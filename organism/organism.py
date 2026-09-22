"""The organism: assembles modules around the thalamic bus and runs the closed
perception -> cognition -> action loop, continuously (real clock) or step-by-step (virtual clock).
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any, Callable

from .config import MODULE_NAMES, Settings
from .core.bus import Bus
from .core.clock import Clock, TemporalState
from .core.events import (
    CommandPayload, DetectedObject, Event, EventType, Modality, SystemPayload, VisionData,
)
from .core.module import CognitiveModule, Context
from .core.trace import TransitionRecorder
from .core.util import new_id
from .models.embeddings import HashEmbedder, build_embedder
from .models.llm import LLMRouter
from .modules.action import Action
from .modules.attention import Attention
from .modules.audition import Audition
from .modules.brainstem import Brainstem
from .modules.emotion import Emotion
from .modules.expression import Expression
from .modules.goals import Goals
from .modules.imagination import Imagination
from .modules.knowledge import Knowledge
from .modules.language import Language
from .modules.memory import Memory
from .modules.metacognition import Metacognition
from .modules.prediction import Prediction
from .modules.safety import Safety
from .modules.self_model import SelfModel
from .modules.sleep import Sleep
from .modules.speech import Speech
from .modules.thought import Thought
from .modules.vision import Vision
from .modules.working_memory import WorkingMemory
from .modules.world_model import WorldModel
from .modules.workspace import LocalRelay, Workspace
from .persistence.store import Store

log = logging.getLogger("organism")

MODULE_CLASSES: dict[str, type[CognitiveModule]] = {
    "brainstem": Brainstem, "attention": Attention, "workspace": Workspace, "working_memory": WorkingMemory,
    "memory": Memory, "emotion": Emotion, "goals": Goals, "thought": Thought, "language": Language,
    "speech": Speech, "action": Action, "safety": Safety, "prediction": Prediction, "imagination": Imagination,
    "sleep": Sleep, "self_model": SelfModel, "world_model": WorldModel,
    "metacognition": Metacognition, "vision": Vision, "audition": Audition, "expression": Expression,
    "knowledge": Knowledge,
}
assert set(MODULE_CLASSES) == set(MODULE_NAMES)


class Organism:
    def __init__(self, settings: Settings, offline_seconds: float = 1.0):
        """``offline_seconds``: virtual clock only - how much organism time passed while it was off."""
        self.settings = settings
        self.offline_seconds = offline_seconds
        self.modules: dict[str, CognitiveModule] = {}
        self.ctx: Context | None = None
        self.bus: Bus | None = None
        self.store: Store | None = None
        self.started = False
        self._flush_task: asyncio.Task | None = None
        self._speech_waiters: list[asyncio.Future] = []

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> "Organism":
        s = self.settings
        s.data_dir.mkdir(parents=True, exist_ok=True)
        self.store = Store(s.data_dir / "organism.db", run_label=s.run_label, redact_pii=s.trace.redact_pii)
        lifecycle = self.store.kv_get("organism.lifecycle", {}) or {}
        start_time = None
        if s.clock.mode == "virtual":
            last = max(lifecycle.get("last_time") or 0.0, lifecycle.get("last_shutdown") or 0.0)
            start_time = (last + self.offline_seconds) if last else None
        clock = Clock(s.clock.mode, s.clock.time_scale, start=start_time)
        now = clock.now()

        lifecycle.setdefault("organism_id", new_id("org_"))
        lifecycle.setdefault("created_at", now)
        lifecycle["boot_count"] = int(lifecycle.get("boot_count", 0)) + 1
        last_shutdown = lifecycle.get("last_shutdown") or lifecycle.get("last_time")
        downtime = (now - last_shutdown) if last_shutdown else None
        lifecycle["last_boot"] = now
        self.store.kv_set("organism.lifecycle", lifecycle)

        temporal = TemporalState(boot_time=now, current_time=now, previous_time=now, last_shutdown=last_shutdown,
                                 downtime=downtime)
        temporal.new_episode(now)
        llm = LLMRouter(s.llm)
        await llm.initialize()
        embedder = await build_embedder(s.embeddings, s.llm)
        recorder = TransitionRecorder(self.store, s.trace.record_transitions, s.trace.capture_state)
        self.bus = Bus(clock, self.store, recorder)
        enabled = {m for m in MODULE_NAMES if s.enabled(m)}
        self.ctx = Context(settings=s, bus=self.bus, clock=clock, store=self.store, llm=llm, embedder=embedder,
                           fast_embedder=HashEmbedder(s.embeddings.hash_dims), temporal=temporal,
                           rng=random.Random(s.seed), enabled_modules=enabled)

        for name in MODULE_NAMES:
            if name in enabled:
                self.modules[name] = MODULE_CLASSES[name](self.ctx)
            elif name == "workspace":
                self.modules["workspace_bypass"] = LocalRelay(self.ctx)
        for m in self.modules.values():
            self.bus.attach(m.name, m.handle, [str(t) for t in m.subscriptions], snapshot=m.trace_state)
        for m in self.modules.values():
            await m.start()
        self.bus.add_observer(self._observe)
        await self.bus.start()
        self.started = True

        self.bus.emit(EventType.SYSTEM_BOOT, "organism",
                      SystemPayload(info={"boot_count": lifecycle["boot_count"], "downtime_s": downtime,
                                          "llm": llm.name, "embedder": embedder.name,
                                          "modules": sorted(self.modules), "disabled": sorted(set(MODULE_NAMES) - enabled)}),
                      summary=(f"I started up (boot #{lifecycle['boot_count']})"
                               + (f" after {downtime / 60:.1f} minutes offline" if downtime else " for the first time")),
                      nominated=True, salience=0.6, modality=Modality.INTEROCEPTIVE)
        if not clock.virtual:
            self._flush_task = asyncio.create_task(self._flush_loop(), name="store-flush")
        log.info("organism started: llm=%s embedder=%s modules=%d", llm.name, embedder.name, len(self.modules))
        return self

    async def stop(self) -> None:
        if not self.started:
            return
        self.bus.emit(EventType.SYSTEM_SHUTDOWN, "organism", SystemPayload(info={}), summary="shutting down")
        try:
            await self.bus.settle(timeout=10)
        except TimeoutError:
            log.warning("shutdown: cognition did not settle")
        for m in reversed(list(self.modules.values())):
            try:
                await m.stop()
            except Exception:
                log.exception("module %s failed to stop", m.name)
        if self._flush_task:
            self._flush_task.cancel()
        await self.bus.stop()
        now = self.ctx.clock.now()
        lc = self.store.kv_get("organism.lifecycle", {}) or {}
        lc["last_shutdown"] = now
        lc["last_time"] = now
        lc["total_uptime_s"] = lc.get("total_uptime_s", 0.0) + (now - self.ctx.temporal.boot_time)
        self.store.kv_set("organism.lifecycle", lc)
        await self.ctx.llm.close()
        await self.ctx.embedder.close()
        self.store.close()
        self.started = False

    async def _flush_loop(self) -> None:
        while True:
            await asyncio.sleep(1.0)
            try:
                self.store.flush()
                lc = self.store.kv_get("organism.lifecycle", {}) or {}
                lc["last_time"] = self.ctx.clock.now()   # survives crashes (downtime estimate)
                self.store.kv_set("organism.lifecycle", lc)
            except Exception:
                log.exception("flush failed")

    async def __aenter__(self) -> "Organism":
        return await self.start()

    async def __aexit__(self, *exc) -> None:
        await self.stop()

    # ------------------------------------------------------------------ stepping (virtual clock)
    @property
    def settle_timeout(self) -> float:
        return 60.0 if self.ctx.llm.is_symbolic else 900.0

    async def tick(self, n: int = 1) -> None:
        """Advance the organism by ``n`` cognitive cycles (virtual clock) and wait for quiescence."""
        bs: Brainstem | None = self.modules.get("brainstem")  # type: ignore[assignment]
        for _ in range(n):
            await self.bus.settle(self.settle_timeout)
            if self.ctx.clock.virtual:
                self.ctx.clock.advance(self.settings.clock.virtual_tick_seconds)
            bs.tick()
            await self.bus.settle(self.settle_timeout)

    async def advance(self, seconds: float) -> None:
        """Let organism time pass without cognitive cycles (e.g. the organism is left alone)."""
        self.ctx.clock.advance(seconds)

    async def settle(self) -> None:
        await self.bus.settle(self.settle_timeout)

    # ------------------------------------------------------------------ senses
    def _audition(self) -> Audition:
        mod = self.modules.get("audition")
        if mod is None:
            raise RuntimeError("the auditory/text sensory channel is disabled")
        return mod  # type: ignore[return-value]

    async def hear(self, text: str, speaker: str = "user", channel: str = "console") -> Event:
        aud = self.modules.get("audition")
        if aud is not None:
            ev = aud.ingest_text(text, speaker, channel)
        else:  # the typed console is a separate sensor: it survives an auditory lesion
            from .core.events import PerceptPayload, TextData
            ev = self.bus.emit(EventType.PERCEPTION, "console",
                               PerceptPayload(modality=Modality.TEXT, sensor=channel,
                                              description=f'{speaker} wrote: "{text}"',
                                              text=TextData(text=text, speaker=speaker, channel=channel)),
                               summary=f'{speaker} wrote: "{text[:140]}"', modality=Modality.TEXT,
                               salience=0.7, confidence=0.99)
        if self.ctx.clock.virtual:
            await self.bus.settle(self.settle_timeout)
        return ev

    async def hear_audio(self, data: bytes, speaker: str | None = None) -> Event | None:
        return await self._audition().ingest_audio_bytes(data, speaker)

    def hear_sound(self, label: str, loudness_db: float = -30.0) -> Event:
        return self._audition().ingest_sound_event(label, loudness_db)

    async def see(self, image: bytes, sensor: str = "uploaded image", stream: bool = False) -> Event | None:
        """``stream=True`` for live video frames (tracking + hysteresis); False for a single deliberate image."""
        vis: Vision | None = self.modules.get("vision")  # type: ignore[assignment]
        if vis is None:
            raise RuntimeError("the visual system is disabled")
        return await vis.ingest_image(image, sensor=sensor, stream=stream)

    def see_objects(self, labels: list[str], confidence: float = 0.85, scene: str | None = None) -> Event:
        """Inject a structured visual percept (simulated environment / experiments)."""
        vis: Vision | None = self.modules.get("vision")  # type: ignore[assignment]
        if vis is None:
            raise RuntimeError("the visual system is disabled")
        n = max(1, len(labels))
        objs = [DetectedObject(label=l, confidence=confidence, bbox=(i / n, 0.3, (i + 0.8) / n, 0.8))
                for i, l in enumerate(labels)]
        from .modules.vision import spatial_relations
        data = VisionData(objects=objs, people=sum(1 for l in labels if l == "person"), scene=scene,
                          spatial_relations=spatial_relations(objs), change_score=0.5)
        return vis.publish_percept(data, sensor="simulated camera")

    def command(self, command: str, **args: Any) -> Event:
        return self.bus.emit(EventType.COMMAND, "operator", CommandPayload(command=command, args=args),
                             summary=f"operator command: {command}")

    # ------------------------------------------------------------------ conversation helper
    def _observe(self, event: Event) -> None:
        if event.type == EventType.SPEECH_GENERATED:
            for fut in self._speech_waiters:
                if not fut.done():
                    fut.set_result(event)
            self._speech_waiters.clear()

    async def converse(self, text: str, speaker: str = "user", max_ticks: int = 8, timeout: float = 120.0) -> str | None:
        """Say something to the organism and return what it says back (None = it stayed silent)."""
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._speech_waiters.append(fut)
        await self.hear(text, speaker)
        if self.ctx.clock.virtual:
            for _ in range(max_ticks):
                await self.tick()
                if fut.done():
                    break
        try:
            ev = await asyncio.wait_for(fut, timeout=0.01 if self.ctx.clock.virtual else timeout)
            return ev.payload.get("text")
        except asyncio.TimeoutError:
            if fut in self._speech_waiters:
                self._speech_waiters.remove(fut)
            return None

    # ------------------------------------------------------------------ observability
    def snapshot(self) -> dict:
        now = self.ctx.clock.now()
        mods = {name: _safe_snapshot(m) for name, m in self.modules.items()}
        return {
            "time": now, "wall_time": time.time(), "cycle": self.bus.cycle,
            "temporal": self.ctx.temporal.model_dump(),
            "llm": self.ctx.llm.snapshot(), "embedder": self.ctx.embedder.name,
            "enabled": sorted(self.modules), "disabled": sorted(set(MODULE_NAMES) - set(self.modules)),
            "bus": {"published": self.bus.stats.published, "requests": self.bus.stats.requests,
                    "modules": self.bus.module_stats(), "transitions": self.bus.recorder.count if self.bus.recorder else 0},
            "modules": mods,
        }

    def add_observer(self, fn: Callable[[Event], None]) -> None:
        self.bus.add_observer(fn)

    def remove_observer(self, fn: Callable[[Event], None]) -> None:
        self.bus.remove_observer(fn)


def _safe_snapshot(m: CognitiveModule) -> dict:
    try:
        return m.snapshot()
    except Exception as exc:  # observability must never crash cognition
        return {"error": repr(exc)}
