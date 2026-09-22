"""Module-level behaviour: attention, working memory, brainstem, language rules."""
from organism.core.events import EventType, Mode, WorkspaceItem, WorkspacePayload, make_event
from organism.modules.attention import Attention
from organism.modules.language_rules import compose_reply, parse_utterance, state_report
from organism.modules.working_memory import WMItem, WorkingMemory
from tests.conftest import make_organism, run


def test_attention_is_not_just_the_latest_item(tmp_path):
    async def scenario():
        org = await make_organism(tmp_path)
        att: Attention = org.modules["attention"]
        att.pool.clear()
        urgent = make_event(EventType.BODY_STATE, "brainstem", {"mode": "AWAKE"}, summary="alarm: smoke detected",
                            salience=0.8, urgency=0.9, nominated=True)
        trivial = make_event(EventType.BODY_STATE, "brainstem", {"mode": "AWAKE"}, summary="a minor noise",
                             salience=0.05, nominated=True)
        att.pool[urgent.id] = (urgent, org.bus.cycle)
        att.pool[trivial.id] = (trivial, org.bus.cycle)   # the most recent candidate
        result = att.compete(org.bus.cycle, Mode.AWAKE)
        ids = [s.event_id for s in result.selected]
        assert ids and ids[0] == urgent.id and trivial.id not in ids
        assert att.threshold(Mode.ASLEEP) > att.threshold(Mode.AWAKE)   # sensory gating in sleep
        await org.stop()

    run(scenario())


def test_working_memory_is_capacity_limited_and_decays(tmp_path):
    async def scenario():
        org = await make_organism(tmp_path)
        wm: WorkingMemory = org.modules["working_memory"]
        wm.slots.clear()
        now = org.ctx.clock.now()
        for i in range(12):
            wm.insert(WMItem(event_id=f"e{i}", content=f"item {i}", kind="percept", source="t", modality="text",
                             activation=0.2 + i * 0.05, importance=0.0, created=now, refreshed=now))
        assert len(wm.slots) == wm.cfg.capacity
        assert "e0" not in {s.event_id for s in wm.slots}          # weakest evicted
        before = max(s.activation for s in wm.slots)
        await org.advance(wm.cfg.half_life_s)
        wm.decay()
        assert max(s.activation for s in wm.slots) < before * 0.6
        await org.stop()

    run(scenario())


def test_brainstem_arousal_sleep_and_wake(tmp_path):
    async def scenario():
        org = await make_organism(tmp_path)
        bs = org.modules["brainstem"]
        a0 = bs.body.arousal
        await org.hear("Hello there, how are you?")
        await org.tick(1)
        assert bs.body.arousal > a0                                  # phasic arousal from a social stimulus
        org.command("sleep")
        await org.tick(1)
        assert bs.body.mode in (Mode.ASLEEP, Mode.DREAMING)
        assert org.bus.sensory_gain["*"] < 1.0
        await org.hear("Wake up please, are you there?")              # urgent addressed speech wakes it
        await org.tick(1)
        assert bs.body.mode == Mode.AWAKE
        await org.stop()

    run(scenario())


def test_language_parsing():
    a = parse_utterance("Hello, my name is Ada and I love astronomy.", "user")
    assert {(f.relation, f.object) for f in a.facts} >= {("name", "Ada"), ("loves", "astronomy")}
    q = parse_utterance("What do you see?", "user")
    assert q.intent == "question" and q.asks_about == "perception_vision"
    c = parse_utterance("Please go to sleep now", "user")
    assert c.intent == "command" and c.command == "sleep"
    overheard = parse_utterance("the weather is nice today", "bob", "CortexAI", channel="microphone")
    assert not overheard.addressed_to_self


def test_verbal_report_is_derived_from_state():
    text, snap = state_report("arousal", None, {"arousal": 0.42, "energy": 0.9, "fatigue": 0.1, "mode": "AWAKE"})
    assert "0.42" in text and snap["body"]["arousal"] == 0.42
    a = parse_utterance("What is my name?", "user")
    assert compose_reply("answer", a, {})["knowledge_gap"]


def test_workspace_payload_roundtrip():
    item = WorkspaceItem(event_id="e", event_type=EventType.PERCEPTION, source="vision", modality="vision",
                         summary="s", score=0.5, entered_cycle=1)
    p = WorkspacePayload(cycle=1, items=[item], new_item_ids=["e"])
    assert WorkspacePayload.model_validate(p.model_dump(mode="json")).items[0].event_id == "e"


def test_knowledge_is_only_consulted_for_impersonal_questions():
    from organism.modules.knowledge import impersonal_question
    assert impersonal_question("What is a black hole?") == "What is a black hole"
    assert impersonal_question("Can you tell me about the Roman empire?") == "the Roman empire"
    for personal in ("Why did you look up?", "What is my name?", "What is in front of you right now?",
                     "Do you have eyes?"):
        assert impersonal_question(personal) is None
    a = parse_utterance("Tell me about volcanoes")
    assert a.intent == "question" and a.asks_about == "general"
    book = {"known": True, "answer": "A volcano is an opening in a planet's crust.", "confidence": 0.8}
    r = compose_reply("answer", a, {"knowledge": book})
    assert r["text"] == "From what I've read, a volcano is an opening in a planet's crust." and r["grounding"] == ["knowledge"]


def test_thoughts_are_told_as_speech_not_as_internal_logs():
    from organism.modules.language_rules import speakable
    from organism.modules.thought import rule_thought
    raw = ("I notice: That was unexpected (My speak action succeeded (I expected p=0.60 of success)). "
           "My model of the situation needs updating; I should look into it.")
    said = speakable(raw)
    assert "(" not in said and "I notice" not in said and "p=0.60" not in said and not said.endswith("…")
    inv = rule_thought("investigation", raw, {}, 0, {}).thought
    assert inv.startswith("Thinking it over:") and "(" not in inv and "I notice" not in inv
    long = "word " * 80
    assert len(speakable(long)) <= 163 and speakable(long).endswith("...")
    a = parse_utterance("what are you thinking right now")
    assert a.asks_about == "thinking"
    r = compose_reply("answer", a, {"thoughts": [{"content": raw}]})
    assert r["text"].startswith("I was just thinking: That was unexpected.") and "(" not in r["text"]


def test_mind_wandering_wonders_about_things_not_words():
    from organism.modules.thought import book_sentence, wonder_topic
    assert wonder_topic('user said to me: "I just bought a telescope to watch the planets"') == "telescope"
    assert wonder_topic("I was surprised: that happened unexpectedly") is None      # its own logs
    assert wonder_topic('user said to me: "it went unexpectedly well"') is None      # no thing to wonder about
    assert wonder_topic('user said to me: "hi, I\'m Radouane"') is None             # names aren't in books
    assert book_sentence("Unexpectedly refers to something occurring in an unforeseen manner.", "unexpectedly") is None
    long = ("A telescope is an optical instrument that gathers light, using lenses or curved mirrors, to make "
            "distant objects appear larger, brighter and closer to the observer than they really are.")
    said = book_sentence(long, "telescope")
    assert said.startswith("A telescope is an optical instrument") and len(said.split()) <= 22 and "…" not in said
    from organism.modules.thought import recollection, worth_wandering_to
    assert not worth_wandering_to("I thought: Thinking it over: user's tone became more positive than I expected.")
    assert not worth_wandering_to("I was surprised: Unexpected speech after a long silence from user.")
    assert not worth_wandering_to('I said to user: "Got it."')
    assert worth_wandering_to('user said to me: "I just bought a telescope"')
    assert not worth_wandering_to('user said to me: "what are you thinking right now?"')
    assert recollection("The user's name is Radouane.") == "I'm remembering: your name is Radouane."
    r = recollection("user's tone became warmer. Then something else happened later on.")
    assert "you's" not in r and "your tone" in r and "later on" not in r
