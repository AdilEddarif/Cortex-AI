"""Temporal memory: forgetting curve, spacing effect, gist, episodes and recall by time."""
from datetime import datetime

from tests.conftest import make_organism, run
from organism.modules.temporal_memory import (
    Timeline, gist, initial_stability, retention, stability_after_recall, time_label, time_reference,
)


def test_forgetting_curve_and_spacing_effect():
    base = 6 * 3600
    weak = initial_stability("episodic", 0.2, 0.0, base)
    vivid = initial_stability("episodic", 0.8, 0.8, base)
    fact = initial_stability("semantic", 0.2, 0.0, base)
    assert weak < vivid < fact * 2 and fact == weak * 8        # important/emotional and knowledge last longer
    assert retention(0, weak) == 1.0
    assert retention(3600, weak) > retention(86400, weak) > retention(3 * 86400, weak)
    assert retention(3 * 86400, weak) < 0.01                   # an unimportant episode is gone in days
    massed = stability_after_recall(weak, retention_at_recall=0.95)
    spaced = stability_after_recall(weak, retention_at_recall=0.3)
    assert weak < massed < spaced                              # recalling when half-forgotten helps most


def test_gist_keeps_the_meaning_not_the_words():
    g = gist('user said to me: "I just bought a telescope to watch the planets tonight"')
    assert g.startswith("user talked to me about") and "telescope" in g and "tonight" not in g
    assert gist('user said to me: "ok"') == "user said something to me."
    assert gist("I decided to express because user asked me to express; outcome: success.") == "I decided to express."


def test_time_references_and_labels():
    now = datetime(2026, 9, 22, 15, 0).timestamp()
    y = time_reference("what did we talk about yesterday evening?", now)
    assert y["label"] == "yesterday evening" and datetime.fromtimestamp(y["start"]).hour == 17
    assert time_reference("what happened before you fell asleep?", now)["anchor"] == "before_sleep"
    two = time_reference("what did I say two hours ago", now)
    assert two["start"] < now - 2 * 3600 < two["end"]
    assert time_reference("what is a telescope?", now) is None
    assert time_label(now - 30, now) == "just now"
    assert time_label(datetime(2026, 9, 22, 9, 0).timestamp(), now) == "this morning"
    assert time_label(datetime(2026, 9, 21, 19, 0).timestamp(), now) == "yesterday evening"


def test_timeline_segments_and_summarises_episodes():
    tl = Timeline()
    tl.observe("ep1", "episodic", 'user said to me: "I bought a telescope"', 0.5, 100.0, ["user"])
    tl.observe("ep1", "episodic", 'user said to me: "the telescope shows planets"', 0.5, 110.0, ["user"])
    tl.observe("ep1", "episodic", "I decided to express because user asked me to express; outcome: success.", 0.3, 120.0, [])
    tl.close("sleep")
    tl.observe("ep2", "episodic", 'user said to me: "good morning"', 0.4, 5000.0, ["user"])
    first = tl.episodes[0]
    assert first.end_reason == "sleep" and first.summary.startswith("I talked with user about the telescope")
    assert "made faces" in first.summary and "went to sleep" in first.summary
    assert tl.before_last_sleep() is first and tl.after_last_sleep() is tl.episodes[1]
    assert tl.between(0, 200) == [first]
    restored = Timeline(tl.to_list())                          # after a restart, open episodes are closed
    assert restored.episodes[-1].end_reason == "shutdown"


def test_faded_memories_need_a_strong_cue_and_fade_to_gist_during_sleep(tmp_path):
    async def scenario():
        org = await make_organism(tmp_path)
        await org.tick(1)
        mem = org.modules["memory"]
        now = org.ctx.clock.now()
        old = await mem.system.add("episodic", 'user said to me: "my old bicycle has a squeaky blue bell"',
                                   now - 3 * 86400, importance=0.2,
                                   context={"stability_s": initial_stability("episodic", 0.2, 0, 21600)})
        fresh = await mem.system.add("episodic", 'user said to me: "my new bicycle is red"', now, importance=0.2)
        assert mem.system.retention(old, now) < 0.01 < mem.system.retention(fresh, now)
        hits = [r.id for r, _s, _rel in await mem.system.search("bicycle", now=now, touch=False)]
        assert fresh.id in hits and old.id not in hits          # a faded memory doesn't come back on a weak cue
        assert await mem._fade({}) >= 1                          # during sleep: details go, the gist stays
        assert old.content.startswith("user talked to me about") and "squeaky" not in old.content
        assert old.context["gist"] and fresh.content.endswith('red"')
        await org.stop()

    run(scenario())


def test_recall_by_time_after_sleep_and_the_next_day(tmp_path):
    async def scenario():
        org = await make_organism(tmp_path)
        await org.tick(2)
        await org.converse("I just bought a telescope to watch the planets")
        await org.converse("The telescope makes Saturn look amazing")
        await org.converse("Go to sleep now.")
        await org.tick(10)
        await org.hear("Wake up please!")
        await org.tick(2)
        reply = await org.converse("What did we talk about before you fell asleep?") or ""
        assert "telescope" in reply.lower(), reply
        eps = org.modules["memory"].timeline.episodes
        assert any(e.end_reason == "sleep" for e in eps)
        await org.advance(86400)                                 # a day passes
        await org.tick(1)
        reply = await org.converse("What did we talk about yesterday?") or ""
        assert "telescope" in reply.lower() and reply.lower().startswith("yesterday"), reply
        reply = await org.converse("What did we do last week?") or ""
        assert reply
        await org.stop()

    run(scenario())


def test_temporal_memory_can_be_ablated(tmp_path):
    async def scenario():
        org = await make_organism(tmp_path)
        org.settings.memory.episodes = False
        mem = org.modules["memory"]
        mem.cfg.episodes = False
        mem.system.forgetting = False
        await org.tick(1)
        await org.converse("I love telescopes")
        reply = await org.converse("What did we talk about yesterday?") or ""
        assert "can't place things in time" in reply
        now = org.ctx.clock.now()
        rec = await mem.system.add("episodic", "an old memory", now - 30 * 86400, importance=0.2)
        assert mem.system.retention(rec, now) == 1.0
        await org.stop()

    run(scenario())
