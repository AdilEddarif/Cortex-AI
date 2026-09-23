"""Reasoning: answering what no single source can answer, and explaining its own behaviour.

The cortex has typed sources — memory, the self-model, the world model, prediction, its books and
the web — each of which answers one kind of question. Reasoning is the step above: it recognises the
*shape* of a question, plans a short chain of lookups over those sources, combines the results and
reports the chain it followed. The plan is the cortex's own (deterministic rules, so it is
reproducible and ablatable); the language model is never asked to reason, only to translate an
utterance (``comprehension``) or to recall what it read (``knowledge``).

Shapes it handles:
  * **multi-hop** — "who is the president of the country where the Eiffel Tower is?"
    (find the country, then who holds the office there)
  * **comparison** — "which is taller, X or Y?" (look up both, compare the numbers, state both)
  * **time arithmetic** — "how many days until X?" (its own clock plus a looked-up date)
  * **self-explanation** — "why did you say that?" (walk the real provenance chain of its own events)

If a step fails, it says which one failed. Nothing is filled in by guessing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime

from ..core.events import Event, EventType, Modality, ThoughtPayload
from ..core.module import CognitiveModule
from ..core.util import new_id, truncate

# ---------------------------------------------------------------------------------- question shapes
MULTI_HOP = re.compile(
    r"^(?:who|what)\s+(?:is|are|was)\s+(?:the\s+)?(?P<office>president|prime minister|king|queen|capital|currency|"
    r"population|leader|head of state)\s+of\s+the\s+(?P<via>country|city|place|nation|state)\s+"
    r"(?:where|in which|that)\s+(?P<anchor>.+?)\s*(?:is|are|lies|stands|sits)?\s*\??$", re.I)
COMPARISON = re.compile(
    r"^(?:which|who)\s+(?:one\s+)?(?:is|was)\s+(?P<adj>taller|higher|bigger|larger|longer|older|heavier|deeper|"
    r"faster|smaller|shorter|younger|lighter|newer)\s*[,:]?\s*(?P<a>.+?)\s+or\s+(?P<b>.+?)\s*\??$", re.I)
TIME_UNTIL = re.compile(r"^how\s+(?:many\s+days|long)\s+(?P<dir>until|till|to|since)\s+(?P<what>.+?)\s*\??$", re.I)
WHY_DID = re.compile(r"^why\s+(?:did|do|are|were|is)\s+you\s+(?P<what>.+?)\s*\??$", re.I)

ATTRIBUTE = {"taller": "height", "higher": "height", "bigger": "size", "larger": "size", "longer": "length",
             "older": "age", "heavier": "weight", "deeper": "depth", "faster": "speed", "smaller": "size",
             "shorter": "height", "younger": "age", "lighter": "weight", "newer": "age"}
SMALLER_WINS = {"smaller", "shorter", "younger", "lighter", "newer"}
# Units brought to one scale so two looked-up numbers can be compared at all.
UNITS = {"mm": 0.001, "cm": 0.01, "m": 1.0, "metre": 1.0, "metres": 1.0, "meter": 1.0, "meters": 1.0,
         "km": 1000.0, "ft": 0.3048, "feet": 0.3048, "foot": 0.3048, "mi": 1609.34, "mile": 1609.34,
         "miles": 1609.34, "kg": 1.0, "kilograms": 1.0, "tonnes": 1000.0, "tons": 907.18, "years": 1.0}
_NUMBER = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(mm|cm|km|m|metres|metre|meters|meter|ft|feet|foot|miles|mile|mi|"
                     r"kg|kilograms|tonnes|tons|years)?\b", re.I)
_DATE = re.compile(r"\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b|\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b|\b(\d{4})-(\d{2})-(\d{2})\b|\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\b|\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})\b", re.I)
_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
           "november", "december"]
_PLACE = re.compile(r"\bin\s+(?:the\s+)?([A-Z][\w'-]*(?:[\s,]+[A-Z][\w'-]*)*)")


@dataclass
class Chain:
    question: str
    shape: str
    steps: list[dict] = field(default_factory=list)   # {step, source, result}
    answer: str = ""
    known: bool = False
    confidence: float = 0.0
    failed_at: str = ""

    def step(self, what: str, source: str, result: str) -> None:
        self.steps.append({"step": what, "source": source, "result": truncate(result, 160)})

    def as_text(self) -> str:
        return "; ".join(f"{s['step']} -> {s['result']}" for s in self.steps)


def number_in(text: str) -> tuple[float, str] | None:
    """The first magnitude in a sentence, converted to a common scale ("324 metres" -> 324.0 m)."""
    for m in _NUMBER.finditer(text):
        raw, unit = m.group(1).replace(",", ""), (m.group(2) or "").lower()
        try:
            value = float(raw)
        except ValueError:
            continue
        if len(raw) == 4 and not unit and 1000 <= value <= 2999:
            continue                      # a year, not a measurement
        return value * UNITS.get(unit, 1.0), unit or ""
    return None


def date_in(text: str, today: date | None = None) -> date | None:
    """A date in a sentence. "December 25" has no year: the next such date from today is meant."""
    m = _DATE.search(text)
    if not m:
        return None
    today = today or date.today()
    if m.group(1):
        return date(int(m.group(3)), _MONTHS.index(m.group(2).lower()) + 1, int(m.group(1)))
    if m.group(4):
        return date(int(m.group(6)), _MONTHS.index(m.group(4).lower()) + 1, int(m.group(5)))
    if m.group(7):
        return date(int(m.group(7)), int(m.group(8)), int(m.group(9)))
    day, month = (m.group(10), m.group(11)) if m.group(10) else (m.group(13), m.group(12))
    when = date(today.year, _MONTHS.index(month.lower()) + 1, int(day))
    return when if when >= today else date(today.year + 1, when.month, when.day)


def place_in(text: str) -> str | None:
    """The country or city a sentence places something in ("... is in Paris, France." -> "France")."""
    best = None
    for m in _PLACE.finditer(text):
        best = m.group(1).split(",")[-1].strip() or best
    if best:
        return best.rstrip(".")
    caps = re.findall(r"\b[A-Z][a-z]{3,}\b", text)
    return caps[-1] if caps else None


class Reasoning(CognitiveModule):
    name = "reasoning"
    subscriptions = ("*",)

    def __init__(self, ctx):
        super().__init__(ctx)
        self.log: dict[str, Event] = {}          # recent events by id, for walking provenance
        self.order: list[str] = []
        self.recent: list[dict] = []
        self.stats = {"asked": 0, "answered": 0, "failed": 0, "explained": 0}

    async def start(self) -> None:
        self.respond("reason.answer", self.answer)
        self.respond("reason.explain", self.explain)

    async def handle(self, event: Event) -> None:
        self.log[event.id] = event
        self.order.append(event.id)
        if len(self.order) > 400:
            for eid in self.order[:100]:
                self.log.pop(eid, None)
            self.order = self.order[100:]

    # ------------------------------------------------------------------ what shape is this question?
    @staticmethod
    def shape_of(question: str) -> str | None:
        q = question.strip()
        for name, pattern in (("multi_hop", MULTI_HOP), ("comparison", COMPARISON), ("time_until", TIME_UNTIL),
                              ("why", WHY_DID)):
            if pattern.match(q):
                return name
        return None

    # ------------------------------------------------------------------ answering
    async def answer(self, q: dict) -> dict:
        question = str(q.get("question", "")).strip()
        shape = self.shape_of(question)
        if not shape:
            return {"known": False, "shape": None}
        self.stats["asked"] += 1
        chain = Chain(question=question, shape=shape)
        await {"multi_hop": self._multi_hop, "comparison": self._comparison,
               "time_until": self._time_until, "why": self._why}[shape](chain)
        self.stats["answered" if chain.known else "failed"] += 1
        self.recent = ([{"t": self.now(), "question": question, "shape": shape, "known": chain.known,
                         "answer": chain.answer, "steps": chain.steps}] + self.recent)[:12]
        if chain.steps:
            self.emit(EventType.THOUGHT_GENERATED,
                      ThoughtPayload(content=f"Working it out: {chain.as_text()}", kind="inference",
                                     confidence=chain.confidence, chain_id=new_id("c_"), trigger="reasoning"),
                      summary=f"(reasoning) {truncate(chain.as_text(), 140)}", modality=Modality.INTERNAL,
                      nominated=True, salience=0.4, confidence=chain.confidence)
        return {"known": chain.known, "answer": chain.answer, "steps": chain.steps, "shape": shape,
                "confidence": round(chain.confidence, 3), "failed_at": chain.failed_at}

    async def _ask_books(self, question: str, want: str | None = None) -> dict:
        return await self.ask_one("knowledge.query", {"question": question, "want": want}, default=None) or {}

    async def _multi_hop(self, chain: Chain) -> None:
        m = MULTI_HOP.match(chain.question)
        office, anchor = m.group("office").lower(), m.group("anchor").strip()
        first = await self._ask_books(f"Where is {anchor}", want="place")
        if not first.get("known"):
            chain.failed_at = f"where {anchor} is"
            chain.answer = f"I couldn't find out where {anchor} is, so I can't work the rest out."
            return
        chain.step(f"where is {anchor}", first.get("source", "books"), first["answer"])
        where = place_in(first["answer"])
        if not where:
            chain.failed_at = "the place in that answer"
            chain.answer = f"I found out about {anchor}, but couldn't pin down which country that is."
            return
        chain.step("the country", "reasoning", where)
        second = await self._ask_books(f"Who is the {office} of {where}" if office not in ("capital", "currency",
                                                                                           "population")
                                       else f"What is the {office} of {where}")
        if not second.get("known"):
            chain.failed_at = f"the {office} of {where}"
            chain.answer = f"{anchor} is in {where}, but I couldn't find out the {office} of {where}."
            return
        chain.step(f"the {office} of {where}", second.get("source", "books"), second["answer"])
        chain.known, chain.answer = True, second["answer"]
        chain.confidence = min(float(first.get("confidence", 0.6)), float(second.get("confidence", 0.6)))

    async def _comparison(self, chain: Chain) -> None:
        m = COMPARISON.match(chain.question)
        adj, a, b = m.group("adj").lower(), m.group("a").strip(" ,"), m.group("b").strip(" ,")
        attribute = ATTRIBUTE.get(adj, "size")
        values = {}
        for name in (a, b):
            found = await self._ask_books(f"What is the {attribute} of {name}", want="number")
            if not found.get("known"):
                chain.failed_at = f"the {attribute} of {name}"
                chain.answer = f"I couldn't find the {attribute} of {name}, so I can't compare them."
                return
            number = number_in(found["answer"])
            if not number:
                chain.failed_at = f"a number for {name}"
                chain.answer = f"I read about {name}, but found no {attribute} I could compare."
                return
            values[name] = (number[0], found["answer"], found.get("confidence", 0.6))
            chain.step(f"the {attribute} of {name}", found.get("source", "books"), found["answer"])
        winner = min(values, key=lambda k: values[k][0]) if adj in SMALLER_WINS else max(
            values, key=lambda k: values[k][0])
        other = b if winner == a else a
        chain.known = True
        chain.confidence = min(v[2] for v in values.values())
        chain.answer = (f"{winner} is {adj} than {other}." if values[a][0] != values[b][0]
                        else f"{a} and {b} are about the same.")

    async def _time_until(self, chain: Chain) -> None:
        m = TIME_UNTIL.match(chain.question)
        direction, what = m.group("dir").lower(), m.group("what").strip()
        found = await self._ask_books(f"On what date is {what}" if direction != "since" else f"When was {what}",
                                      want="date")
        if not found.get("known"):
            chain.failed_at = f"the date of {what}"
            chain.answer = f"I couldn't find the date of {what}."
            return
        chain.step(f"the date of {what}", found.get("source", "books"), found["answer"])
        today = datetime.fromtimestamp(self.now()).date()
        when = date_in(found["answer"], today)     # "December 25" means the next one by its own clock
        if not when:
            chain.failed_at = "a date in that answer"
            chain.answer = f"I read about {what}, but found no date I could count from."
            return
        days = (when - today).days if direction != "since" else (today - when).days
        chain.step("my clock", "self", today.isoformat())
        chain.known, chain.confidence = True, float(found.get("confidence", 0.7))
        if days < 0:
            chain.answer = (f"{what} was on {when.isoformat()}, {abs(days)} days ago." if direction != "since"
                            else f"{what} is on {when.isoformat()}, {abs(days)} days from now.")
        else:
            chain.answer = (f"{days} days: {what} is on {when.isoformat()}." if direction != "since"
                            else f"{days} days: {what} was on {when.isoformat()}.")

    async def _why(self, chain: Chain) -> None:
        what = WHY_DID.match(chain.question).group("what").strip()
        story = self.explain_action(what)
        if not story:
            chain.failed_at = "an action of mine that matches"
            chain.answer = ""
            return
        chain.steps.extend(story["steps"])
        chain.known, chain.answer, chain.confidence = True, story["answer"], 0.8
        self.stats["explained"] += 1

    # ------------------------------------------------------------------ explaining itself
    async def explain(self, q: dict) -> dict:
        story = self.explain_action(str(q.get("about", "")))
        return story or {"known": False}

    def explain_action(self, about: str) -> dict | None:
        """Walk the provenance of its own recent events: what led to what it just did."""
        words = {w for w in re.findall(r"[a-z']+", about.lower()) if len(w) > 2}
        matches = []
        for eid in reversed(self.order):
            e = self.log.get(eid)
            if e is None or e.type not in (EventType.SPEECH_GENERATED, EventType.ACTION_EXECUTED):
                continue
            text = (e.summary or "").lower() + " " + str(e.payload.get("text", "")).lower()
            if not words or words & set(re.findall(r"[a-z']+", text)) or about.lower() in ("that", "it", "this"):
                matches.append(e)
            if len(matches) >= 6:
                break
        # the event that recorded *why* it acted explains better than the words it spoke
        target = next((e for e in matches if ((e.payload.get("spec") or {}).get("reason"))), None) or             (matches[0] if matches else None)
        if target is None:
            return None
        steps, seen, cursor, reason = [], set(), target, ""
        spec = target.payload.get("spec") or {}
        if spec.get("reason"):
            reason = str(spec["reason"])
        while cursor is not None and len(steps) < 4:
            steps.append({"step": "because", "source": cursor.source, "result": cursor.summary})
            nxt = None
            for cid in cursor.caused_by or []:
                if cid in self.log and cid not in seen:
                    seen.add(cid)
                    nxt = self.log[cid]
                    break
            cursor = nxt
        trigger = steps[-1]["result"] if steps else ""
        answer = f"Because {reason[0].lower() + reason[1:]}" if reason else f"It came from: {trigger}"
        if trigger and reason:
            answer += f". It started with: {trigger}."
        return {"known": True, "answer": answer.strip(". ") + ".", "steps": steps[::-1]}

    def snapshot(self) -> dict:
        return {"stats": dict(self.stats), "recent": self.recent, "events_tracked": len(self.log)}
