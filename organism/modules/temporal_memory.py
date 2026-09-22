"""Temporal memory: how memories live in time (forgetting, fading detail, episodes, "when").

Pure functions and a small timeline class, used by the memory module. Human-memory findings
they model:

* **Forgetting curve** (Ebbinghaus): retention falls off exponentially with time since a memory
  was last used, ``R = exp(-t / S)``. The stability ``S`` is larger for important and emotional
  memories, and for knowledge (semantic, procedural) than for episodes.
* **Spacing effect / desirable difficulty**: each retrieval resets the curve and increases the
  stability, more so when the memory was already half-forgotten than when it was fresh.
* **Accessibility**: a faded memory still exists but only comes back with a strong cue.
* **Gist over verbatim** (fuzzy-trace theory): as an episode fades, its exact words are lost and
  only the gist remains ("you talked to me about telescopes"). Very important memories keep
  their detail longer (flashbulb memories).
* **Event segmentation**: experience is cut into episodes at pauses, sleep and big surprises.
  Each episode keeps who was there, what it was about and what happened, and can be recalled
  by time ("yesterday evening", "before you fell asleep").
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta

from ..core.util import content_words, new_id, truncate

# Relative stability of each kind of memory (episodes fade fastest, skills slowest).
KIND_STABILITY = {"episodic": 1.0, "dream": 0.3, "imagined": 0.5, "autobiographical": 6.0, "semantic": 8.0,
                  "procedural": 20.0}

# ---------------------------------------------------------------------------------- forgetting curve


def initial_stability(kind: str, importance: float, emotion_intensity: float, base_s: float) -> float:
    return base_s * (0.4 + 1.2 * importance + 0.8 * emotion_intensity) * KIND_STABILITY.get(kind, 1.0)


def retention(age_since_use_s: float, stability_s: float) -> float:
    """Probability-like retention in [0, 1] of a memory last used ``age_since_use_s`` ago."""
    return math.exp(-max(0.0, age_since_use_s) / max(1.0, stability_s))


def stability_after_recall(stability_s: float, retention_at_recall: float) -> float:
    """Spacing effect: recalling a half-forgotten memory strengthens it more than a fresh one."""
    return stability_s * (1.2 + 1.8 * (1.0 - retention_at_recall))


# ---------------------------------------------------------------------------------- gist

# Words that carry no topic (function words of the organism's own logs and of conversation).
_NOT_TOPICS = {"said", "decided", "because", "outcome", "success", "user", "asked", "tell", "know", "think",
               "want", "like", "just", "really", "please", "okay", "yeah", "hello", "thanks", "thank", "sure",
               "good", "nice", "great", "what", "your", "you", "can", "could", "would", "will", "something",
               "anything", "nothing", "thing", "things", "look", "right", "now", "today", "time", "maybe",
               "name", "remember", "doing", "going", "make", "made", "much", "very", "also", "well", "mean",
               # talk about the conversation itself, and time words
               "talk", "talked", "sleep", "asleep", "before", "after", "yesterday", "evening", "morning",
               "afternoon", "night", "hours", "minutes", "earlier", "first", "happened", "said"}
_COMMAND = re.compile(r"^(please\s+)?(smile|wink|laugh|frown|grin|look|close|open|shut|count|go|wake|sleep|stop|"
                      r"raise|imagine|remember|write|note|take a note)\b", re.I)
_QUESTION = re.compile(r"(\?\s*$|^(what|who|where|when|why|how|which|do|does|did|can|could|are|is|will|would)\b)", re.I)


_DETERMINERS = {"a", "an", "the", "my", "your", "his", "her", "our", "their", "this", "that", "these", "those",
                "some", "any"}
_PHRASE_END = {"to", "of", "in", "on", "at", "with", "for", "from", "and", "or", "but", "is", "are", "was", "were",
               "has", "have", "had", "makes", "make", "made", "looks", "look", "that", "which", "who", "tonight",
               "today", "now", "yesterday", "tomorrow", "again", "too", "very", "so", "because", "when", "while",
               # common verbs that end a noun phrase ("the telescope | shows planets")
               "shows", "show", "gets", "got", "goes", "went", "sees", "saw", "broke", "works", "needs", "likes",
               "loves", "wants", "can", "will", "would", "could", "should", "did", "does", "do", "isn't", "keeps",
               "seems", "feels", "sounds", "takes", "took", "gave", "gives", "told", "tells", "said", "says"}


def noun_heads(text: str) -> list[str]:
    """Heads of noun phrases ("my old bicycle" -> bicycle): the words a sentence is about."""
    toks = re.findall(r"[a-z']+|[.,!?;]", text.lower())
    heads = []
    for i, w in enumerate(toks):
        if w not in _DETERMINERS:
            continue
        run = []
        for t in toks[i + 1:i + 4]:
            if t in _PHRASE_END or t in _DETERMINERS or not t.isalpha():
                break
            run.append(t)
        if run:
            heads.append(run[-1])
    return heads


def topic_words(text: str, n: int = 3) -> list[str]:
    names = {w.lower() for w in re.findall(r"\b[A-Z][a-z]+\b", text)}

    def ok(w: str) -> bool:
        return w not in _NOT_TOPICS and w not in names and len(w) > 3

    heads = [w for w in noun_heads(text) if ok(w)]
    words = heads or [w for w in content_words(text) if ok(w)]
    return [w for w, _ in Counter(words).most_common(n)]


def _about(words: list[str]) -> str:
    if not words:
        return ""
    words = [w if w.endswith("s") else f"the {w}" for w in words]  # "the telescope", "planets"
    return " about " + (words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1])


def gist(content: str) -> str:
    """The gist that survives when the details of an episode are forgotten."""
    t = content.strip()
    m = re.match(r'^(\w+) said to me: "(.*)"$', t, flags=re.S)
    if m:
        about = _about(topic_words(m.group(2), 2))
        return f"{m.group(1)} talked to me{about}." if about else f"{m.group(1)} said something to me."
    m = re.match(r'^I said to (\w+): "(.*)"$', t, flags=re.S)
    if m:
        about = _about(topic_words(m.group(2), 2))
        return f"I talked to {m.group(1)}{about}." if about else f"I said something to {m.group(1)}."
    m = re.match(r"^I decided to (\w+)", t)
    if m:
        return f"I decided to {m.group(1)}."
    head = re.split(r"[:;(]", t, maxsplit=1)[0].strip().rstrip(".")
    words = head.split()
    return (" ".join(words[:8]) + ("..." if len(words) > 8 else "")).rstrip(".") + "."


# ---------------------------------------------------------------------------------- episodes

ACTION_PHRASES = {"express": "made faces", "sleep": "went to sleep", "simulate": "imagined things",
                  "look": "looked around", "remember": "memorised something", "note": "wrote a note",
                  "set_goal": "took on a goal"}


@dataclass
class Episode:
    id: str
    start: float
    end: float
    n: int = 0
    participants: list[str] = field(default_factory=list)
    topics: dict[str, int] = field(default_factory=dict)
    actions: list[str] = field(default_factory=list)
    moments: list[list] = field(default_factory=list)   # [importance, text] of the most important memories
    end_reason: str | None = None                        # None while the episode is still going on
    summary: str = ""

    def observe(self, kind: str, content: str, importance: float, ts: float, participants: list[str]) -> None:
        self.n += 1
        self.end = max(self.end, ts)
        for p in participants:
            if p and p not in ("self", "me") and p not in self.participants:
                self.participants.append(p)
        m = re.match(r'^(\w+) said to me: "(.*)"$', content, flags=re.S)
        if m:
            said = m.group(2).strip()
            if not (_QUESTION.search(said) or _COMMAND.search(said)):  # what was said, not questions or requests
                for w in topic_words(m.group(2), 4):
                    self.topics[w] = self.topics.get(w, 0) + 1
            if m.group(1) not in self.participants:
                self.participants.append(m.group(1))
        m = re.match(r"^I decided to (\w+)", content)
        if m and m.group(1) in ACTION_PHRASES and ACTION_PHRASES[m.group(1)] not in self.actions:
            self.actions.append(ACTION_PHRASES[m.group(1)])
        if kind == "episodic" and not content.startswith(("I decided to", "I said to")):
            self.moments.append([round(importance, 3), truncate(content, 160)])
            self.moments = sorted(self.moments, key=lambda x: -x[0])[:3]
        self.summary = self.describe()

    def describe(self) -> str:
        parts = []
        topics = [w for w, _ in sorted(self.topics.items(), key=lambda kv: (-kv[1], kv[0]))[:3]]
        people = [p for p in self.participants if p != "self"]
        if people:
            who = people[0] if len(people) == 1 else ", ".join(people[:-1]) + " and " + people[-1]
            parts.append(f"I talked with {who}{_about(topics)}")
        elif topics:
            parts.append(f"I was thinking{_about(topics)}")
        acts = list(self.actions)
        if self.end_reason == "sleep" and "went to sleep" not in acts:
            acts.append("went to sleep")
        if acts:
            acts = acts[:3]
            parts.append("I " + (acts[0] if len(acts) == 1 else ", ".join(acts[:-1]) + " and " + acts[-1]))
        if not parts and self.moments:
            parts.append(gist(self.moments[0][1]).rstrip("."))
        return "; ".join(parts) + "." if parts else "nothing much happened."


class Timeline:
    """Experience cut into episodes. Persisted as plain dicts."""

    def __init__(self, episodes: list[dict] | None = None, cap: int = 300):
        self.cap = cap
        self.episodes: list[Episode] = [Episode(**e) for e in (episodes or [])]
        for e in self.episodes:  # an episode left open by a shutdown ended when its last memory did
            if e.end_reason is None:
                e.end_reason = "shutdown"

    @property
    def current(self) -> Episode | None:
        return self.episodes[-1] if self.episodes and self.episodes[-1].end_reason is None else None

    def observe(self, episode_id: str, kind: str, content: str, importance: float, ts: float,
                participants: list[str]) -> Episode:
        cur = self.current
        if cur is None or cur.id != episode_id:
            if cur is not None:
                self.close("pause")
            cur = Episode(id=episode_id or new_id("ep_"), start=ts, end=ts)
            self.episodes.append(cur)
            self.episodes = self.episodes[-self.cap:]
        cur.observe(kind, content, importance, ts, participants)
        return cur

    def close(self, reason: str) -> Episode | None:
        cur = self.current
        if cur is not None:
            cur.end_reason = reason
            cur.summary = cur.describe()
        return cur

    def between(self, start: float, end: float) -> list[Episode]:
        return [e for e in self.episodes if e.n and e.start <= end and e.end >= start]

    def before_last_sleep(self) -> Episode | None:
        slept = [e for e in self.episodes if e.end_reason == "sleep" and e.n]
        return slept[-1] if slept else None

    def after_last_sleep(self) -> Episode | None:
        for i in range(len(self.episodes) - 1, -1, -1):
            if self.episodes[i].end_reason == "sleep":
                later = [e for e in self.episodes[i + 1:] if e.n]
                return later[0] if later else None
        return None

    def first(self) -> Episode | None:
        with_people = [e for e in self.episodes if e.n and e.participants]
        return with_people[0] if with_people else None

    def to_list(self) -> list[dict]:
        return [asdict(e) for e in self.episodes]


# ---------------------------------------------------------------------------------- when

PARTS_OF_DAY = {"morning": (5, 12), "afternoon": (12, 17), "evening": (17, 22), "night": (22, 29)}


def _day(ts: float) -> datetime:
    d = datetime.fromtimestamp(ts)
    return d.replace(hour=0, minute=0, second=0, microsecond=0)


def _part_of_day(hour: int) -> str:
    for name, (a, b) in PARTS_OF_DAY.items():
        if a <= hour < b or a <= hour + 24 < b:
            return name
    return "night"


def time_label(ts: float, now: float) -> str:
    """How a person would say when something happened ("just now", "this morning", "yesterday evening")."""
    age = now - ts
    if age < 120:
        return "just now"
    if age < 3600:
        m = round(age / 60)
        return f"about {m} minutes ago"
    d, today = _day(ts), _day(now)
    part = _part_of_day(datetime.fromtimestamp(ts).hour)
    if d == today:
        if age < 3 * 3600:
            h = round(age / 3600)
            return "about an hour ago" if h <= 1 else f"about {h} hours ago"
        return "last night" if part == "night" and datetime.fromtimestamp(ts).hour < 5 else f"this {part}"
    if d == today - timedelta(days=1):
        return "last night" if part == "night" else f"yesterday {part}"
    days = (today - d).days
    if days < 7:
        return f"on {d.strftime('%A')}"
    return f"{days} days ago" if days < 30 else d.strftime("on %d %B")


_UNITS = {"second": 1, "minute": 60, "hour": 3600, "day": 86400, "week": 7 * 86400}
_NUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "ten": 10,
        "twenty": 20, "few": 3, "couple of": 2, "couple": 2}


def time_reference(text: str, now: float) -> dict | None:
    """A time reference in a question, as a window {start, end, label} or a special anchor."""
    low = text.lower()
    if re.search(r"before (you|u) (fell asleep|slept|went to sleep|were asleep|napped)|before your (nap|sleep)", low):
        return {"anchor": "before_sleep", "label": "before I fell asleep"}
    if re.search(r"(after|when) (you|u) (woke up|wake up|work up)", low):
        return {"anchor": "after_sleep", "label": "after I woke up"}
    if re.search(r"first (time )?(we|i) (met|talked|spoke)|when we first (met|talked)", low):
        return {"anchor": "first", "label": "when we first met"}
    m = re.search(r"\b(\d+|an?|one|two|three|four|five|six|ten|twenty|few|couple of|couple) "
                  r"(second|minute|hour|day|week)s? ago\b", low)
    if m:
        n = int(m.group(1)) if m.group(1).isdigit() else _NUM.get(m.group(1), 1)
        span = n * _UNITS[m.group(2)]
        return {"start": now - span * 1.6, "end": now - span * 0.5, "label": f"{m.group(0)}"}
    today = _day(now).timestamp()
    yesterday = today - 86400
    if "last night" in low:
        return {"start": yesterday + 18 * 3600, "end": today + 5 * 3600, "label": "last night"}
    for part, (a, b) in PARTS_OF_DAY.items():
        if re.search(rf"\byesterday {part}\b", low):
            return {"start": yesterday + a * 3600, "end": yesterday + b * 3600, "label": f"yesterday {part}"}
        if re.search(rf"\b(this|today) {part}\b", low) or (part == "night" and "tonight" in low):
            return {"start": today + a * 3600, "end": min(now, today + b * 3600), "label": f"this {part}"}
    if "yesterday" in low:
        return {"start": yesterday, "end": today, "label": "yesterday"}
    if re.search(r"\b(earlier today|today)\b", low):
        return {"start": today, "end": now, "label": "earlier today"}
    if re.search(r"\blast week\b", low):
        return {"start": now - 7 * 86400, "end": now, "label": "in the last week"}
    return None
