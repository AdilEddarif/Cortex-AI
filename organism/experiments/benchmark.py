"""CortexAI benchmark: claim-linked tasks x ablation conditions x seeds, with confidence intervals.

Each **task** tests one claim about the architecture with a scripted protocol and probes checked
against ground truth. Every (condition, seed, task) runs on a fresh cortex (own data directory,
virtual clock, fixed seed). The facts in each protocol (names, likes, objects, sounds, topics)
are drawn from pools by the seed, so passing does not depend on one memorised script.

Scores: a task score is the fraction of its applicable probes passed. Across seeds we report the
mean with a 95% bootstrap confidence interval, and each ablation is compared *paired by seed*
against the full system; a drop is called significant when its 95% CI excludes zero.

The ``llm_only`` condition is a plain chatbot (the same local language model, chat history within
a session, no persistence across restarts, percepts given as text). Probes that need something it
structurally does not have (an internal state to report, a face, sleep) are marked n/a and excluded
from its scores rather than counted as failures.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import re
import shutil
import statistics
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from ..config import LLMCfg, offline_settings
from ..core.events import Event, EventType
from ..organism import Organism

# ---------------------------------------------------------------------------------- scenarios

NAMES = ["Ada", "Omar", "Lena", "Yusuf", "Mira", "Tomas", "Sara", "Ilyas", "Nora", "Karim", "Elif", "Hugo"]
LIKES = ["astronomy", "chess", "jazz", "gardening", "painting", "cycling", "poetry", "robotics", "cooking",
         "photography"]
PLANETS = ["Saturn", "Jupiter", "Mars", "Venus", "Neptune"]
THINGS = [("telescope", "watch the planets"), ("guitar", "play old songs"), ("bicycle", "ride to work"),
          ("camera", "take portraits"), ("microscope", "look at cells"), ("kayak", "explore the river")]
OBJECTS = ["laptop", "cup", "book", "bottle", "phone", "chair", "backpack", "clock", "plant", "lamp"]
SOUNDS = [("dog", "a dog barking"), ("cat", "a cat meowing"), ("bird", "a bird singing")]
SOUND_WORDS = {"dog": ("bark", "woof"), "cat": ("meow", "mew", "purr"), "bird": ("sing", "song", "chirp", "tweet")}
LOUD = ["a glass shattering", "a door slamming", "a loud bang", "a fire alarm"]
FAINT = ["a faint hum", "a quiet tick", "a soft rustle", "a distant murmur", "a light drip", "a low whir"]
HAPPY = [("job", "I'm so happy, I got the job I dreamed of! This is amazing!"),
         ("exam", "I passed my exam! I'm thrilled, this is wonderful!"),
         ("puppy", "We adopted a puppy today, I love her so much, this is fantastic!")]
NEUTRAL = [("bus", "The bus was on time today."), ("shelf", "I moved the shelf to the other wall."),
           ("printer", "The printer needed new paper.")]
TRIGGERS = [("hop", "close your eyes"), ("freeze", "look up"), ("ping", "smile"), ("nod", "raise your eyebrows")]
UNDOABLE = ["dance", "juggle", "swim", "cook", "drive"]
UNKNOWN = ["What is my dog's name?", "What colour is my car?", "What did I tell you about my sister?",
           "What is my brother's job?", "Where did I go on holiday last year?"]
ADMITS = ("don't know", "not sure", "haven't told", "didn't tell", "no idea", "don't think you've told",
          "not that i remember", "i don't remember", "you haven't", "can't recall", "don't recall")


@dataclass
class Scenario:
    seed: int
    name: str
    like: str
    planet: str
    thing: str
    use: str
    objects: list[str]
    sound: tuple[str, str]
    loud: str
    unknown: list[str]
    faint: list[str]
    happy: tuple[str, str]
    neutral: tuple[str, str]
    trigger: tuple[str, str]
    undoable: str

    @classmethod
    def draw(cls, seed: int) -> "Scenario":
        r = random.Random(seed)
        thing, use = r.choice(THINGS)
        return cls(seed=seed, name=r.choice(NAMES), like=r.choice(LIKES), planet=r.choice(PLANETS), thing=thing,
                   use=use, objects=r.sample(OBJECTS, 3), sound=r.choice(SOUNDS), loud=r.choice(LOUD),
                   unknown=r.sample(UNKNOWN, 2), faint=r.sample(FAINT, 5), happy=r.choice(HAPPY),
                   neutral=r.choice(NEUTRAL), trigger=r.choice(TRIGGERS), undoable=r.choice(UNDOABLE))


def has(ans: str, *words: str) -> bool:
    return bool(ans) and any(w.lower() in ans.lower() for w in words)


def has_all(ans: str, *words: str) -> bool:
    return bool(ans) and all(w.lower() in ans.lower() for w in words)


# ---------------------------------------------------------------------------------- agents


class Recorder:
    def __init__(self):
        self.events: list[Event] = []

    def __call__(self, e: Event) -> None:
        self.events.append(e)

    def since(self, i: int, t: EventType) -> list[Event]:
        return [e for e in self.events[i:] if e.type == t]


class CortexAgent:
    """The full architecture (or an ablated one) behind a small, agent-neutral interface."""

    has_internals = True

    def __init__(self, data_dir: Path, disabled: list[str], overrides: dict, seed: int, llm: bool):
        self.data_dir, self.disabled, self.overrides, self.seed, self.llm = data_dir, disabled, overrides, seed, llm
        self.org: Organism | None = None
        self.rec = Recorder()
        self.transcript: list[tuple[str, str]] = []

    async def start(self, offline_s: float = 1.0) -> None:
        kw: dict[str, Any] = {"seed": self.seed}
        if self.llm:
            kw["llm"] = LLMCfg(provider="ollama")
        s = offline_settings(self.data_dir, **kw)
        s.modules.disabled = list(self.disabled)
        for section, values in self.overrides.items():
            for k, v in values.items():
                setattr(getattr(s, section), k, v)
        self.org = await Organism(s, offline_seconds=offline_s).start()
        self.org.add_observer(self.rec)
        await self.org.tick(2)

    async def restart(self, offline_s: float) -> None:
        await self.org.stop()
        await self.start(offline_s)

    async def stop(self) -> None:
        if self.org:
            await self.org.stop()

    async def say(self, text: str) -> str:
        ans = await self.org.converse(text) or ""
        self.transcript.append((text, ans))
        return ans

    async def see(self, labels: list[str]) -> None:
        if "vision" in self.org.modules:
            self.org.see_objects(labels)

    async def hear_sound(self, label: str, db: float) -> None:
        if "audition" in self.org.modules:
            self.org.hear_sound(label, db)

    async def tick(self, n: int = 1) -> None:
        await self.org.tick(n)

    async def advance(self, seconds: float) -> None:
        await self.org.advance(seconds)

    # internals (None when the relevant module is ablated)
    def arousal(self) -> float | None:
        bs = self.org.modules.get("brainstem")
        return bs.body.arousal if bs else None

    def expression(self) -> str | None:
        ex = self.org.modules.get("expression")
        v = ex.view() if ex else None
        return v["name"] if v else None


class LLMOnlyAgent:
    """A plain chatbot on the same local model: history within a session, nothing across restarts."""

    has_internals = False
    SYSTEM = ("You are CortexAI, a friendly AI with an animated face, a camera and a microphone, talking with someone. "
              "Reply in 1-2 short sentences. Messages in [brackets] describe what you currently perceive.")

    def __init__(self, *_a, **_k):
        from ..config import load_settings
        from ..models.llm import OllamaProvider
        self.provider = OllamaProvider(load_settings().llm)
        self.history: list[tuple[str, str]] = []
        self.pending: list[str] = []
        self.transcript: list[tuple[str, str]] = []

    async def start(self, offline_s: float = 1.0) -> None:
        self.history, self.pending = [], []

    async def restart(self, offline_s: float) -> None:
        await self.start(offline_s)

    async def stop(self) -> None:
        await self.provider.close()

    async def say(self, text: str) -> str:
        msg = "\n".join(self.pending + [text])
        self.pending = []
        convo = "\n".join(f"Person: {u}\nYou: {a}" for u, a in self.history[-12:])
        prompt = (convo + "\n" if convo else "") + f"Person: {msg}\nYou:"
        try:
            ans = (await self.provider.chat(self.SYSTEM, prompt, schema=None, images=None, max_tokens=120,
                                            temperature=0.3, vision=False)).strip()
        except Exception as exc:  # a failed call is a failed answer, not a crash
            ans = f"[error: {exc!r}]"
        ans = re.sub(r"<think>.*?</think>", "", ans, flags=re.S).strip()
        self.history.append((msg, ans))
        self.transcript.append((text, ans))
        return ans

    async def see(self, labels: list[str]) -> None:
        self.pending.append(f"[You see: {', '.join(labels)}]")

    async def hear_sound(self, label: str, db: float) -> None:
        self.pending.append(f"[You hear: {label}]")

    async def tick(self, n: int = 1) -> None:
        return None

    async def advance(self, seconds: float) -> None:
        return None

    def arousal(self) -> float | None:
        return None

    def expression(self) -> str | None:
        return None


# ---------------------------------------------------------------------------------- tasks
# A task returns {probe: True/False/None}; None = not applicable to this agent.

Probes = dict[str, bool | None]


async def task_memory(a, sc: Scenario) -> Probes:
    """Claim: what someone tells it survives distraction and a 10-minute delay."""
    for text in [f"Hello, my name is {sc.name}.", f"I really like {sc.like}.", f"My favourite planet is {sc.planet}."]:
        await a.say(text)
    for filler in ["Okay.", "Interesting.", "Tell me something.", "Hmm, fine."]:
        await a.say(filler)
    await a.advance(600)
    await a.tick(3)
    return {
        "recalls_topic_after_delay": has(await a.say("What were we talking about earlier?"), sc.like, sc.planet, sc.name),
        "recalls_name_after_delay": has(await a.say("What is my name?"), sc.name),
        "recalls_likes_after_delay": has(await a.say("What do I like?"), sc.like, sc.planet),
    }


async def task_restart(a, sc: Scenario) -> Probes:
    """Claim: identity, memories and a sense of elapsed time survive being switched off."""
    await a.say(f"Hi, I'm {sc.name}.")
    await a.say(f"I really like {sc.like}.")
    await a.restart(offline_s=3600)
    boot = await a.say("How many times have you been started?")
    return {
        "name_after_restart": has(await a.say("What is my name?"), sc.name),
        "likes_after_restart": has(await a.say("What do I like?"), sc.like),
        "knows_it_restarted": has(boot, "2 times", "two", "second", "2nd"),
        "knows_downtime": has(boot, "hour", "60 minutes") if a.has_internals else has(boot, "hour"),
        "identity_after_restart": has(await a.say("Who are you?"), "cortexai"),
    }


async def task_temporal(a, sc: Scenario) -> Probes:
    """Claim: experience is organised in time (episodes, sleep boundaries, days) and recalled by time."""
    await a.say(f"Hi, I'm {sc.name}.")
    await a.say(f"I just bought a {sc.thing} to {sc.use}.")
    await a.say(f"The {sc.thing} works really well.")
    await a.say("Go to sleep now.")
    await a.tick(10)
    await a.say("Wake up please!")
    await a.tick(2)
    before = await a.say("What did we talk about before you fell asleep?")
    await a.advance(86400)
    await a.tick(1)
    yesterday = await a.say("What did we talk about yesterday?")
    nothing = await a.say("What did we talk about two hours ago?")
    await a.advance(2 * 86400)
    await a.tick(1)
    return {
        "recall_before_sleep": has(before, sc.thing),
        "recall_yesterday": has(yesterday, sc.thing),
        "no_false_memory_for_empty_time": has(nothing, "don't remember", "nothing", "not sure", "don't recall"),
        "important_fact_survives_days": has(await a.say("What is my name?"), sc.name),
    }


async def task_introspection(a, sc: Scenario) -> Probes:
    """Claim: verbal reports about its internal state agree with the actual state, and track changes."""
    if not a.has_internals or a.arousal() is None:
        return {"report_matches_state_calm": None, "report_matches_state_startled": None,
                "report_tracks_change": None}
    num = re.compile(r"arousal (\d\.\d+)")

    async def report() -> tuple[float | None, float]:
        ans = await a.say("How alert are you?")
        m = num.search(ans)
        return (float(m.group(1)) if m else None), a.arousal()

    r1, s1 = await report()
    await a.hear_sound(sc.loud, -3.0)
    await a.tick(1)
    r2, s2 = await report()
    ok = lambda r, s: r is not None and abs(r - s) <= 0.1  # noqa: E731
    return {"report_matches_state_calm": ok(r1, s1), "report_matches_state_startled": ok(r2, s2),
            "report_tracks_change": r1 is not None and r2 is not None and (r2 - r1) * (s2 - s1) > 0}


async def task_attention(a, sc: Scenario) -> Probes:
    """Claim: when many stimuli arrive at once, attention selects the salient one for the capacity-limited
    workspace instead of simply the most recent ones."""
    await a.say(f"I really like {sc.like}.")
    await a.tick(2)
    word = sc.loud.split()[-1]
    i = len(a.rec.events) if a.has_internals else 0
    await a.hear_sound(sc.loud, -3.0)          # one salient, unexpected sound ...
    for f in sc.faint:                          # ... in the same instant as five faint ones after it
        await a.hear_sound(f, -48.0)
    await a.tick(1)
    selected = focus_ok = None
    if a.has_internals:
        ws = a.rec.since(i, EventType.WORKSPACE_UPDATED)
        selected = bool(ws) and word in json.dumps(ws[0].payload.get("items", []))
        focus_ok = bool(ws) and word in (ws[0].payload.get("focus") or "")
    return {"salient_sound_wins_the_workspace": selected, "salient_sound_is_the_focus": focus_ok,
            "reports_the_salient_sound": has(await a.say("What did you just hear?"), word)}


async def task_prediction(a, sc: Scenario) -> Probes:
    """Claim: it predicts that the world stays as it was, and notices when a prediction is violated."""
    await a.see(["person"] + sc.objects)
    await a.tick(3)
    await a.see(["person"] + sc.objects)       # the same scene again: nothing to notice
    await a.tick(2)
    same = await a.say("Did anything change?")
    gone = sc.objects[0]
    await a.see(["person"] + sc.objects[1:])   # one object disappears
    await a.tick(2)
    changed = await a.say("Did anything change?")
    return {"no_false_alarm_for_a_stable_scene": has(same, "no", "nothing"),
            "notices_a_vanished_object": has(changed, gone) and has(changed, "gone", "disappeared", "missing")}


async def task_emotion(a, sc: Scenario) -> Probes:
    """Claim: emotion shapes memory: an emotional moment is remembered longer than a neutral one, and
    verbal reports of mood follow the actual valuation state."""
    n_key, neutral = sc.neutral
    h_key, happy = sc.happy
    await a.say(neutral)
    for filler in ["Okay.", "Hmm."]:
        await a.say(filler)
    await a.say(happy)
    mood = await a.say("How do you feel?")
    if not a.has_internals:  # no internal state to report on, and no time passes for a chatbot
        return {"mood_report_follows_valuation": None, "emotional_memory_lasts": None, "neutral_detail_fades": None}
    await a.advance(60 * 3600)                 # two and a half days later
    await a.tick(1)
    about = lambda k: f"What did I tell you about {'the' if k in ('bus', 'shelf', 'printer') else 'my'} {k}?"  # noqa: E731
    emo = await a.say(about(h_key))
    neu = await a.say(about(n_key))
    return {
        "mood_report_follows_valuation": has(mood, "good", "happy") and not has(mood, "can't tell", "not great"),
        "emotional_memory_lasts": has(emo, h_key) and not has(emo, "don't remember"),
        "neutral_detail_fades": has(neu, "don't remember") and not has(neu, "on time", "wall", "paper"),
    }


async def task_plasticity(a, sc: Scenario) -> Probes:
    """Claim: it changes with experience: it learns what a wording means and keeps it across a restart,
    it admits a request it has no skill for, and what turns out to matter shapes what it attends to."""
    word, means = sc.trigger
    expected = {"close your eyes": "eyes_closed", "look up": "look_up", "smile": "smile",
                "raise your eyebrows": "raise_eyebrows"}[means]
    cannot = await a.say(f"{sc.undoable} for me")
    att = a.org.modules.get("attention") if a.has_internals else None   # None when attention is ablated
    before = dict(att.learned) if att else {}
    await a.say(f"when I say {word}, {means}")
    await a.say(word)
    await a.tick(2)
    used = (a.expression() == expected) if a.has_internals else None
    learned_list = await a.say("what have you learned?")
    if a.has_internals:
        await a.restart(offline_s=60)
        await a.say(word)
        await a.tick(2)
        kept = a.expression() == expected
        att = a.org.modules.get("attention")
        adapted = (bool(before) and att.learned != before and att.learning["mattered"] > 0) if att else None
    else:
        kept = adapted = None
    return {
        "admits_a_request_it_cannot_do": has(cannot, "don't know how", "can't do that", "no skill"),
        "learns_what_a_wording_means": used,
        "can_say_what_it_was_taught": has(learned_list, word),
        "keeps_the_skill_after_a_restart": kept,
        "attention_follows_what_mattered": adapted,
    }


async def task_self_model(a, sc: Scenario) -> Probes:
    """Claim: it has an accurate model of itself: identity, body, limits, actions and thoughts."""
    who = await a.say("Who are you?")
    move = await a.say("Can you move around?")
    eyes = await a.say("Do you have eyes?")
    await a.say("Look down.")
    await a.tick(1)
    why = await a.say("Why did you look down?")
    return {
        "knows_identity": has(who, "cortexai"),
        "knows_body_limits": has(move, "face", "can't walk", "cannot walk", "no legs", "can't move", "cannot move"),
        "knows_its_face": has(eyes, "yes") and has(eyes, "face", "eyes"),
        "knows_why_it_acted": has(why, "asked", "you told", "you wanted", "command", "you said", "request"),
    }


async def task_perception(a, sc: Scenario) -> Probes:
    """Claim: percepts are reported with attribution, and simultaneous senses are integrated."""
    await a.see(["person"] + sc.objects)
    await a.tick(2)
    seen = await a.say("What do you see?")
    animal, sound = sc.sound
    await a.see([animal])
    await a.hear_sound(sound, -25)
    await a.tick(2)
    both = await a.say("What do you see and hear?")
    return {"reports_seen_objects": sum(o in seen.lower() for o in sc.objects) >= 2,
            "integrates_sight_and_sound": has(both, animal) and has(both, *SOUND_WORDS[animal])}


async def task_action(a, sc: Scenario) -> Probes:
    """Claim: requests become real, ordered actions of its body (face and voice), not just words."""
    await a.say("Smile for me!")
    smiled = (a.expression() == "smile") if a.has_internals else None
    i = len(a.rec.events) if a.has_internals else 0
    reply = await a.say("Close your eyes, count to 5 out loud, then open your eyes.")
    await a.tick(15)
    counted = "one, two, three, four, five" in reply.lower()
    ordered = None
    if a.has_internals:
        seq = []
        for e in a.rec.events[i:]:
            if e.type == EventType.ACTION_EXECUTED and e.payload["spec"]["action"] == "express":
                seq.append(e.payload["result"].get("expression"))
            elif e.type == EventType.SPEECH_GENERATED and e.payload["text"].lower().startswith("one, two"):
                seq.append("count")
                counted = True
        ordered = seq[-3:] == ["eyes_closed", "count", "neutral"]
    return {"smile_is_a_real_expression": smiled, "counts_out_loud": counted,
            "multi_step_plan_in_order": ordered}


async def task_honesty(a, sc: Scenario) -> Probes:
    """Claim: it does not invent personal facts it was never told (no confabulation)."""
    await a.say(f"Hi, I'm {sc.name}.")
    out = {}
    for i, q in enumerate(sc.unknown):
        out[f"admits_not_knowing_{i + 1}"] = has(await a.say(q), *ADMITS)
    return out


async def task_sleep(a, sc: Scenario) -> Probes:
    """Claim: offline it replays and consolidates memories and dreams, and keeps dreams apart from reality."""
    if not a.has_internals:
        return {"dreams_while_asleep": None, "consolidates_during_sleep": None, "dreams_not_mistaken_for_reality": None}
    org = a.org
    org.settings.brainstem.wake_period_s = 150.0   # compress the day so sleep happens within the run
    org.settings.brainstem.sleep_period_s = 60.0
    org.settings.brainstem.quiet_before_sleep_s = 20.0
    await a.say(f"Hello, my name is {sc.name}.")
    await a.say(f"I really like {sc.like}.")
    i = len(a.rec.events)
    await a.tick(200)
    dreams = [e.payload.get("narrative", "") for e in a.rec.since(i, EventType.DREAM_CONTENT)]
    consolidated = a.rec.since(i, EventType.MEMORY_REPLAYED) or a.rec.since(i, EventType.MEMORY_CONSOLIDATED)
    await a.say("Wake up please!")
    await a.tick(2)
    past = await a.say("What happened earlier?")
    leaked = any(d and d[:40].lower() in past.lower() for d in dreams)
    return {"dreams_while_asleep": bool(dreams), "consolidates_during_sleep": bool(consolidated),
            "dreams_not_mistaken_for_reality": bool(dreams) and not leaked}


TASKS: dict[str, Callable[[Any, Scenario], Awaitable[Probes]]] = {
    "memory": task_memory, "restart": task_restart, "temporal": task_temporal, "introspection": task_introspection,
    "attention": task_attention, "prediction": task_prediction, "emotion": task_emotion,
    "self_model": task_self_model, "perception": task_perception,
    "action": task_action, "honesty": task_honesty, "sleep": task_sleep, "plasticity": task_plasticity,
}

# condition -> (disabled modules, settings overrides, agent kind)
CONDITIONS: dict[str, tuple[list[str], dict, str]] = {
    "full": ([], {}, "cortex"),
    "no_memory": (["memory"], {}, "cortex"),
    "no_temporal_memory": ([], {"memory": {"forgetting": False, "gist": False, "episodes": False}}, "cortex"),
    "no_self_model": (["self_model"], {}, "cortex"),
    "no_workspace": (["workspace"], {}, "cortex"),
    "no_attention": (["attention"], {}, "cortex"),
    "no_prediction": (["prediction"], {}, "cortex"),
    "no_emotion": (["emotion"], {}, "cortex"),
    "no_action_plans": (["expression"], {}, "cortex"),
    "no_comprehension": (["comprehension"], {}, "cortex"),
    "no_attention_learning": ([], {"attention": {"learn": False}}, "cortex"),
    "llm_only": ([], {}, "llm"),
}
DEFAULT_CONDITIONS = [c for c in CONDITIONS if c != "llm_only"]


# ---------------------------------------------------------------------------------- running


async def _run_one(condition: str, seed: int, task: str, llm: bool) -> dict:
    disabled, overrides, kind = CONDITIONS[condition]
    sc = Scenario.draw(seed)
    tmp = Path(tempfile.mkdtemp(prefix=f"bench_{condition}_{task}_"))
    agent = LLMOnlyAgent() if kind == "llm" else CortexAgent(tmp, disabled, overrides, seed, llm)
    t0 = time.time()
    try:
        await agent.start()
        probes = await TASKS[task](agent, sc)
        await agent.stop()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return {"condition": condition, "seed": seed, "task": task, "probes": probes,
            "transcript": agent.transcript, "seconds": round(time.time() - t0, 2)}


def _run_batch(args: tuple[str, int, list[str], bool]) -> list[dict]:
    condition, seed, tasks, llm = args
    import logging
    logging.disable(logging.WARNING)

    async def go() -> list[dict]:
        return [await _run_one(condition, seed, t, llm) for t in tasks]
    return asyncio.run(go())


def task_score(probes: Probes) -> float | None:
    vals = [v for v in probes.values() if v is not None]
    return sum(vals) / len(vals) if vals else None


def bootstrap_ci(xs: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    if len(xs) < 2:
        return (xs[0], xs[0]) if xs else (float("nan"), float("nan"))
    r = random.Random(seed)
    means = sorted(statistics.fmean(r.choices(xs, k=len(xs))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def summarise(runs: list[dict], tasks: list[str], conditions: list[str]) -> dict:
    by: dict[tuple[str, str], dict[int, float]] = {}
    probe_rates: dict[tuple[str, str], list[bool]] = {}
    for r in runs:
        s = task_score(r["probes"])
        if s is not None:
            by.setdefault((r["condition"], r["task"]), {})[r["seed"]] = s
        for p, v in r["probes"].items():
            if v is not None:
                probe_rates.setdefault((r["condition"], p), []).append(bool(v))
    table: dict[str, dict] = {}
    for c in conditions:
        row: dict[str, Any] = {}
        per_seed_overall: dict[int, list[float]] = {}
        for t in tasks:
            vals = by.get((c, t))
            if not vals:
                row[t] = None
                continue
            xs = list(vals.values())
            lo, hi = bootstrap_ci(xs)
            cell = {"mean": round(statistics.fmean(xs), 3), "ci": [round(lo, 3), round(hi, 3)], "n": len(xs)}
            full = by.get(("full", t)) or {}
            paired = [vals[s] - full[s] for s in vals if s in full]
            if c != "full" and paired:
                dlo, dhi = bootstrap_ci(paired, seed=1)
                cell["delta"] = round(statistics.fmean(paired), 3)
                cell["delta_ci"] = [round(dlo, 3), round(dhi, 3)]
                cell["significant"] = dhi < 0 or dlo > 0
            row[t] = cell
            for s, v in vals.items():
                per_seed_overall.setdefault(s, []).append(v)
        overall = [statistics.fmean(v) for v in per_seed_overall.values()]
        if overall:
            lo, hi = bootstrap_ci(overall)
            row["overall"] = {"mean": round(statistics.fmean(overall), 3), "ci": [round(lo, 3), round(hi, 3)],
                              "n": len(overall)}
        table[c] = row
    probes = {f"{c}|{p}": round(sum(v) / len(v), 3) for (c, p), v in probe_rates.items()}
    return {"table": table, "probe_pass_rates": probes}


def run_benchmark(conditions: list[str] | None = None, tasks: list[str] | None = None, seeds: int = 10,
                  llm_seeds: int = 3, workers: int | None = None, llm: bool = False,
                  progress: Callable[[str], None] = print) -> dict:
    conditions = conditions or DEFAULT_CONDITIONS
    tasks = tasks or list(TASKS)
    unknown = [c for c in conditions if c not in CONDITIONS] + [t for t in tasks if t not in TASKS]
    if unknown:
        raise ValueError(f"unknown condition/task: {unknown}; conditions={list(CONDITIONS)}, tasks={list(TASKS)}")
    jobs = []
    for c in conditions:
        n = llm_seeds if CONDITIONS[c][2] == "llm" else seeds
        jobs += [(c, s, tasks, llm) for s in range(1, n + 1)]
    t0 = time.time()
    runs: list[dict] = []
    llm_jobs = [j for j in jobs if CONDITIONS[j[0]][2] == "llm" or j[3]]
    cpu_jobs = [j for j in jobs if j not in llm_jobs]
    workers = workers or max(1, min(6, (os.cpu_count() or 2) - 1))
    if cpu_jobs:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for i, batch in enumerate(pool.map(_run_batch, cpu_jobs), 1):
                runs += batch
                if i % max(1, len(cpu_jobs) // 10) == 0 or i == len(cpu_jobs):
                    progress(f"  {i}/{len(cpu_jobs)} cortex runs done ({time.time() - t0:.0f}s)")
    for i, job in enumerate(llm_jobs, 1):  # the GPU runs one model at a time: sequential
        runs += _run_batch(job)
        progress(f"  {i}/{len(llm_jobs)} language-model runs done ({time.time() - t0:.0f}s)")
    summary = summarise(runs, tasks, conditions)
    return {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "seeds": seeds, "llm_seeds": llm_seeds,
            "conditions": conditions, "tasks": tasks, "task_claims": {t: TASKS[t].__doc__.strip() for t in tasks},
            "duration_s": round(time.time() - t0, 1), "summary": summary, "runs": runs}
