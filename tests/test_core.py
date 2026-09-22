"""Unit tests: event schemas, the thalamic bus, persistence, redaction."""
import asyncio

import numpy as np
import pytest

from organism.core.bus import Bus
from organism.core.clock import Clock
from organism.core.events import (
    Event, EventType, Modality, PerceptPayload, TextData, ThoughtPayload, make_event,
)
from organism.core.redact import redact_text
from organism.models.embeddings import HashEmbedder
from organism.modules.memory import MemorySystem
from organism.persistence.store import Store
from tests.conftest import run


def test_payload_schema_is_enforced():
    with pytest.raises(TypeError):
        make_event(EventType.PERCEPTION, "x", ThoughtPayload(content="wrong payload type"))
    with pytest.raises(Exception):
        make_event(EventType.PERCEPTION, "x", {"modality": "text"})  # missing required fields
    ev = make_event(EventType.PERCEPTION, "x", PerceptPayload(modality=Modality.TEXT, sensor="s", description="d",
                                                              text=TextData(text="hi")), confidence=0.8)
    assert ev.uncertainty == pytest.approx(0.2)
    assert ev.data(PerceptPayload).text.text == "hi"


def test_bus_routing_provenance_requests_and_gating():
    async def scenario():
        bus = Bus(Clock("virtual"))
        got_a, got_b = [], []

        async def handler_a(e: Event):
            got_a.append(e)
            if e.type == EventType.COMMAND:
                bus.emit(EventType.MODULE_ERROR, "a", {"module": "a", "error": "child"})

        async def handler_b(e: Event):
            got_b.append(e)

        bus.attach("a", handler_a, [EventType.COMMAND.value])
        bus.attach("b", handler_b, ["*"])

        async def responder(q):
            return {"echo": q["x"]}

        bus.register_responder("topic.echo", "b", responder)
        await bus.start()
        root = bus.emit(EventType.COMMAND, "operator", {"command": "go"})
        await bus.settle()
        assert got_a and got_a[0].id == root.id
        child = next(e for e in got_b if e.type == EventType.MODULE_ERROR)
        assert root.id in child.caused_by                       # provenance
        assert not any(e.source == "b" for e in got_b)          # no self-delivery
        assert await bus.request_one("topic.echo", {"x": 3}) == {"echo": 3}
        assert await bus.request("missing.topic") == []
        bus.sensory_gain["*"] = 0.3                              # asleep: thalamic gating
        p = bus.emit(EventType.PERCEPTION, "vision",
                     PerceptPayload(modality=Modality.VISION, sensor="c", description="d"), salience=0.9,
                     modality=Modality.VISION)
        assert p.salience == pytest.approx(0.27) and p.raw_salience == pytest.approx(0.9)
        await bus.stop()

    run(scenario())


def test_bus_isolates_module_failures():
    async def scenario():
        bus = Bus(Clock("virtual"))
        errors = []

        async def broken(_e):
            raise RuntimeError("boom")

        async def watcher(e):
            if e.type == EventType.MODULE_ERROR:
                errors.append(e)

        bus.attach("broken", broken, [EventType.COMMAND.value])
        bus.attach("watch", watcher, [EventType.MODULE_ERROR.value])
        await bus.start()
        bus.emit(EventType.COMMAND, "op", {"command": "x"})
        await bus.settle()
        assert errors and errors[0].payload["module"] == "broken"
        await bus.stop()

    run(scenario())


def test_memory_system_retrieval_and_persistence(tmp_path):
    async def scenario():
        store = Store(tmp_path / "m.db")
        emb = HashEmbedder(384)
        ms = MemorySystem(store, emb)
        await ms.add("semantic", "The user's name is Ada.", 1000.0, importance=0.8)
        await ms.add("episodic", "I saw a red cup on the table.", 1001.0, importance=0.5)
        await ms.add("dream", "In a dream the user's name was a planet.", 1002.0, importance=0.9)
        hits = await ms.search("what is the user's name", now=1010.0)
        assert hits and hits[0][0].content == "The user's name is Ada."
        assert all(r.kind != "dream" for r, _, _ in hits)        # reality monitoring: dreams excluded by default
        dream_hits = await ms.search("user's name", kinds=["dream"], now=1010.0)
        assert dream_hits and dream_hits[0][0].kind == "dream"
        store.close()
        store2 = Store(tmp_path / "m.db")
        ms2 = MemorySystem(store2, emb)
        assert await ms2.load() == 3
        store2.close()

    run(scenario())


def test_redaction():
    text = "my key is sk-ant-abcdefghijklmnopqrstuv and password: hunter2"
    out = redact_text(text)
    assert "sk-ant" not in out and "hunter2" not in out


def test_hash_embedder_similarity():
    e = HashEmbedder(256)
    a, b, c = e.embed_sync("I love astronomy"), e.embed_sync("astronomy is what I love"), e.embed_sync("red car")
    assert float(np.dot(a, b)) > float(np.dot(a, c))
