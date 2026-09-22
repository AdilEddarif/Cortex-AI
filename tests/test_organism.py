"""Integration tests for the organism's architectural guarantees."""
from organism.core.events import EventType, Mode
from organism.organism import MODULE_CLASSES
from tests.conftest import FakeProvider, make_organism, run


class Log:
    def __init__(self, org):
        self.events = []
        org.add_observer(self.events.append)

    def of(self, t):
        return [e for e in self.events if e.type == t]


def test_learns_remembers_and_survives_restart(tmp_path):
    async def scenario():
        org = await make_organism(tmp_path)
        await org.tick(2)
        assert "Ada" in (await org.converse("Hi, my name is Ada.") or "")
        assert "Ada" in (await org.converse("What is my name?") or "")
        await org.stop()
        org = await make_organism(tmp_path, offline_seconds=7200)
        await org.tick(2)
        assert "Ada" in (await org.converse("What is my name?") or "")
        reply = await org.converse("How many times have you been started?") or ""
        assert "2 times" in reply and "2.0 hours" in reply           # identity + temporal continuity
        await org.stop()

    run(scenario())


def test_silence_is_a_choice(tmp_path):
    async def scenario():
        org = await make_organism(tmp_path)
        log = Log(org)
        await org.tick(2)
        await org.hear("the weather is nice today", speaker="bob", channel="microphone")
        await org.tick(3)
        assert not log.of(EventType.SPEECH_GENERATED)
        waits = [e for e in log.of(EventType.ACTION_SELECTED) if e.payload["spec"]["action"] == "wait"]
        assert waits and "not to me" in waits[0].payload["spec"]["reason"]
        await org.stop()

    run(scenario())


def test_thought_never_becomes_speech_directly(tmp_path):
    async def scenario():
        org = await make_organism(tmp_path)
        log = Log(org)
        await org.tick(2)
        await org.converse("Hello! I am feeling great today.")
        await org.converse("What do you see?")
        thoughts = log.of(EventType.THOUGHT_GENERATED)
        speech = log.of(EventType.SPEECH_GENERATED)
        approved = {e.payload["decision_id"] for e in log.of(EventType.ACTION_APPROVED)}
        assert thoughts and speech
        assert all(s.payload["decision_id"] in approved for s in speech)   # speech only via approved actions
        thought_texts = {t.payload["content"] for t in thoughts}
        assert not any(s.payload["text"] in thought_texts for s in speech)
        await org.stop()

    run(scenario())


def test_verbal_reports_cannot_write_internal_state(tmp_path):
    # Architectural guarantee: no module subscribes to VERBAL_REPORT, and reports are never nominated,
    # so a claim about a state cannot flow back into that state.
    for cls in MODULE_CLASSES.values():
        subs = cls.__dict__.get("subscriptions", ())
        assert isinstance(subs, property) or EventType.VERBAL_REPORT not in subs

    async def scenario():
        org = await make_organism(tmp_path)
        log = Log(org)
        await org.tick(2)
        reply = await org.converse("How alert are you?") or ""
        reports = log.of(EventType.VERBAL_REPORT)
        assert reports and not any(r.nominated for r in reports)
        snap_arousal = reports[0].payload["state_snapshot"]["body"]["arousal"]
        assert f"{snap_arousal:.2f}" in reply
        await org.stop()

    run(scenario())


def test_safety_layer_gates_actions(tmp_path):
    async def scenario():
        org = await make_organism(tmp_path)
        log = Log(org)
        await org.tick(2)
        reply = await org.converse("Please move to the left.") or ""
        assert "just a face" in reply or "only a face" in reply
        await org.converse("Please write down: buy milk")
        rejected = {e.payload["spec"]["action"]: e.payload["reason"] for e in log.of(EventType.ACTION_REJECTED)}
        executed = {e.payload["spec"]["action"] for e in log.of(EventType.ACTION_EXECUTED)}
        # Moving is either foreseen as impossible by imagination (never attempted) or blocked by safety.
        assert "move" not in executed and ("move" not in rejected or "body" in rejected["move"])
        assert "permission" in rejected.get("note", "")
        assert not (tmp_path / "notebook.md").exists()
        from organism.core.events import ActionPayload, ActionSpec, ActionType
        blocked = org.modules["safety"].check(ActionPayload(decision_id="d", spec=ActionSpec(action=ActionType.MOVE),
                                                            status="selected"))
        assert blocked and "body" in blocked
        # grant the permission explicitly
        org.settings.safety.permissions["digital_write"] = True
        await org.converse("Please write down: buy bread")
        assert "buy bread" in (tmp_path / "notebook.md").read_text(encoding="utf-8")
        await org.stop()

    run(scenario())


def test_sleep_consolidates_and_dreams_are_tagged(tmp_path):
    async def scenario():
        org = await make_organism(tmp_path)
        log = Log(org)
        await org.tick(2)
        for text in ["Hi, I'm Ada.", "I love astronomy.", "I love telescopes too.", "Saturn has rings."]:
            await org.converse(text)
        org.command("sleep")
        await org.tick(140)                                             # a requested sleep lasts >= 90 s
        assert org.modules["brainstem"].body.mode == Mode.AWAKE        # woke up rested
        assert log.of(EventType.DREAM_CONTENT) and log.of(EventType.MEMORY_REPLAYED)
        consolidated = log.of(EventType.MEMORY_CONSOLIDATED)
        assert consolidated and consolidated[-1].payload["replayed"] > 0
        mem = org.modules["memory"].system
        assert mem.counts()["dream"] > 0
        hits = await mem.search("astronomy telescopes", now=org.ctx.clock.now(), touch=False)
        assert all(r.kind != "dream" for r, _, _ in hits)
        reply = await org.converse("Did you dream?") or ""
        assert "wasn't real" in reply
        await org.stop()

    run(scenario())


def test_ablations_run_and_change_behaviour(tmp_path):
    async def scenario():
        org = await make_organism(tmp_path / "a", disabled=["memory", "self_model"])
        await org.tick(2)
        await org.converse("My name is Ada.")
        assert "not sure" in (await org.converse("Who are you?") or "")
        await org.stop()
        org = await make_organism(tmp_path / "a", disabled=["memory", "self_model"])
        await org.tick(2)
        assert "told me your name" in (await org.converse("What is my name?") or "")
        await org.stop()
        org = await make_organism(tmp_path / "c", disabled=["workspace"])
        log = Log(org)
        await org.tick(2)
        assert await org.converse("Hello!")
        assert "workspace_bypass" in org.modules
        assert all(not e.payload["integrated"] for e in log.of(EventType.WORKSPACE_UPDATED))
        await org.stop()

    run(scenario())


def test_secrets_are_redacted_from_the_log_and_transitions_recorded(tmp_path):
    async def scenario():
        org = await make_organism(tmp_path)
        await org.tick(1)
        await org.converse("my api_key=sk-abcdefghijklmnopqrstuvwxyz123456")
        org.store.flush()
        dump = "\n".join(str(r) for r in org.store.conn.execute("SELECT summary, payload FROM events").fetchall())
        assert "sk-abcdefghijklmnop" not in dump and "[REDACTED]" in dump
        assert org.store.count("transitions") > 0
        await org.stop()

    run(scenario())


def test_neural_path_and_graceful_degradation(tmp_path):
    async def scenario():
        org = await make_organism(tmp_path)
        fake = FakeProvider()
        org.ctx.llm.provider = fake
        await org.tick(1)
        # The model is book knowledge, consulted for general questions; the organism says it in its own words.
        reply = await org.converse("What is a black hole?") or ""
        assert reply.startswith("From what I've read") and "a neural thought" in reply
        assert fake.prompts and "black hole" in fake.prompts[-1]
        n = len(fake.prompts)
        # Everything about itself, its body, the speaker or the moment is the organism's own business.
        for text in ("I'm upset, my telescope broke.", "Look down.", "What are you thinking?", "Can you move?"):
            reply = await org.converse(text) or ""
            assert reply != "Neural reply." and "a neural thought" not in reply
        assert "Looking down" in (await org.converse("Look down.") or "")
        assert len(fake.prompts) == n
        assert set(org.ctx.llm.stats.by_task) == {"knowledge_lookup"}  # it never thinks, deliberates or speaks
        org.ctx.llm.provider = FakeProvider(fail=True)                # the model server dies
        reply = await org.converse("What is a quasar?")
        assert reply and org.ctx.llm.stats.failures > 0               # it just doesn't know; cognition continues
        await org.stop()

    run(scenario())


def test_expressions_and_requested_sleep_are_real_actions(tmp_path):
    async def scenario():
        org = await make_organism(tmp_path)
        log = Log(org)
        await org.tick(2)
        reply = await org.converse("Smile for me!") or ""
        assert "software" not in reply.lower() and "can't" not in reply.lower()
        assert org.modules["expression"].view()["name"] == "smile"
        await org.converse("Close your eyes.")
        assert org.modules["expression"].view()["name"] == "eyes_closed"
        await org.converse("Go to sleep now.")
        assert org.modules["brainstem"].body.mode in (Mode.ASLEEP, Mode.DREAMING)
        await org.tick(20)                                               # no one talks: it stays asleep
        assert org.modules["brainstem"].body.mode in (Mode.ASLEEP, Mode.DREAMING)
        selected = [e.payload["spec"]["action"] for e in log.of(EventType.ACTION_SELECTED)]
        assert "express" in selected and "sleep" in selected
        await org.hear("Wake up please!")
        await org.tick(1)
        assert org.modules["brainstem"].body.mode == Mode.AWAKE
        await org.stop()

    run(scenario())


def test_multi_step_requests_run_in_order(tmp_path):
    async def scenario():
        org = await make_organism(tmp_path)
        log = Log(org)
        await org.tick(2)
        reply = await org.converse("i want you to close your eyes, count to 10 out loud then open your eyes") or ""
        assert "relax" not in reply.lower()
        await org.tick(25)
        done = []
        for e in log.events:
            if e.type == EventType.ACTION_EXECUTED and e.payload["spec"]["action"] == "express":
                done.append(e.payload["result"]["expression"])
            elif e.type == EventType.SPEECH_GENERATED and e.payload["text"].startswith("One, two"):
                done.append("count")
        assert done == ["eyes_closed", "count", "neutral"], done
        said = [e.payload["text"] for e in log.of(EventType.SPEECH_GENERATED)]
        assert "One, two, three, four, five, six, seven, eight, nine, ten." in said
        view = org.modules["expression"].view()
        assert view is None or view["name"] == "neutral"   # eyes are open again
        assert org.modules["action"].plan is None
        await org.stop()

    run(scenario())
