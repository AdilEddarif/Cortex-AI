"""Book knowledge (semantic cortex fed by reading). The only cognitive role of the language model.

The organism thinks, decides, feels, perceives and speaks with its own modules. The language
model is treated like everything the organism has ever *read*: a large store of general
knowledge it can consult, the way a well-read person recalls something from a book. It is
never the speaker, never the inner voice and never the self:

  * it answers impersonal, encyclopedic questions only ("what is a black hole?");
  * questions about the organism, the speaker or the current situation are refused before the
    model is called: those belong to the self-model, memory, perception and interoception;
  * answers come back as data (known / answer / confidence), which the organism's own language
    system puts into words ("From what I've read, ...");
  * what was looked up is remembered as semantic memory with source "reading".

It can also **look things up** (``models/research.py``): when its books don't know, or when the
question is time-sensitive and book knowledge is likely out of date, a research agent searches
Wikipedia and (with a Tavily key) the web, and answers only from what it fetched, with sources.
This needs the safety permission ``internet_read`` (off by default); the same impersonal filter
applies, so nothing about the organism or the person ever leaves the machine.
"""
from __future__ import annotations

import re
import time

from pydantic import BaseModel, Field

from ..core.module import CognitiveModule
from ..core.util import clamp, content_words, truncate
from ..models.research import ResearchAgent
from ..models.web import WebTools

LIBRARY_SYSTEM = """You are a reference library: the contents of many books. You are not a person, you are not
in a conversation, and nobody is talking to you. Given a question, return general encyclopedic knowledge
in at most two short, plain sentences, written impersonally: never use "I", "me", "you" or "we".
If general knowledge cannot answer it (it depends on a particular person, a private situation, the
present moment, or opinions), set known=false and leave answer empty."""

# Polite frames around a factual question ("can you tell me about ...") are stripped before checking
# whether what remains is personal.
_FRAMES = re.compile(
    r"^(please\s+)?((can|could|would|will) you\s+)?(please\s+)?"
    r"(tell me( more)?( about)?|explain( to me)?|describe|do you know( anything)?( about)?|"
    r"what do you know about|have you (ever )?(heard|read) (of|about))\s*", re.I)
_PERSONAL = re.compile(r"\b(you|your|yours|yourself|me|my|mine|myself|i|i'm|we|us|our|ours)\b", re.I)
_SITUATIONAL = re.compile(r"\b(here|tonight|right now|this room|in front|weather|outside|nearby|near me|local)\b", re.I)
# Questions whose answer changes over time: book knowledge is likely out of date for these.
_TIME_SENSITIVE = re.compile(
    r"\b(latest|current(ly)?|now|today|this (year|month|week)|recent(ly)?|newest|so far|as of|"
    r"20[2-9]\d|who (is|are) the (president|prime minister|ceo|king|queen|champion|leader|head)|"
    r"price of|how much (is|does)|score|won the|winner of|release date|still alive)\b", re.I)


def time_sensitive(question: str) -> bool:
    return bool(_TIME_SENSITIVE.search(question))


RECITE_SYSTEM = """You are a library of things people have written and told each other. Given a kind of piece
(a joke, riddle, poem, story, quote, tongue twister) and optionally a topic, return ONE short, well-known,
family-friendly piece of that kind, as it is usually told. Plain text only, at most 45 words, no preamble,
no explanation, nothing about yourself and nobody addressed as "I" or "you" outside the piece itself."""

# Pieces for the symbolic layer, so a cortex without a language model can still tell one (public domain).
STOCK = {
    "joke": ["Why don't scientists trust atoms? Because they make up everything.",
             "Why did the scarecrow win an award? He was outstanding in his field.",
             "What do you call a fish with no eyes? Fsh."],
    "riddle": ["What has keys but opens no locks? A piano.",
               "What gets wetter the more it dries? A towel."],
    "quote": ["\\u201cThe only thing we have to fear is fear itself.\\u201d",
              "\\u201cI think, therefore I am.\\u201d"],
    "tongue twister": ["She sells seashells by the seashore."],
    "poem": ["Roses are red, violets are blue, memory is fragile, and so are you."],
    "story": ["A hare mocked a tortoise for being slow, so they raced. The hare stopped to nap; the tortoise "
              "kept walking and won."],
}


class Piece(BaseModel):
    text: str = ""


class Lookup(BaseModel):
    known: bool = False
    answer: str = ""
    confidence: float = Field(0.5, ge=0, le=1)


def impersonal_question(text: str) -> str | None:
    """The general-knowledge core of a question, or None if it is about someone or something present."""
    q = _FRAMES.sub("", text.strip()).strip(" ?.!")
    if not q or _PERSONAL.search(q) or _SITUATIONAL.search(q) or not content_words(q):
        return None
    return q


class Knowledge(CognitiveModule):
    name = "knowledge"

    def __init__(self, ctx):
        super().__init__(ctx)
        self.cfg = self.settings.knowledge
        self._cache: dict[str, dict] = {}
        self.recent: list[dict] = []
        self.stats = {"queries": 0, "refused_personal": 0, "known": 0, "unknown": 0, "looked_up": 0}
        self.tools: WebTools | None = None
        self.agent: ResearchAgent | None = None

    async def start(self) -> None:
        self.respond("knowledge.query", self.query)
        self.respond("knowledge.recite", self.recite)
        self.tools = WebTools(lang=self.cfg.wikipedia_lang, timeout_s=self.cfg.timeout_s,
                              cache_ttl_s=self.cfg.cache_ttl_s, max_calls_per_hour=self.cfg.max_web_calls_per_hour)
        self.agent = ResearchAgent(self.ctx.llm, self.tools, max_steps=self.cfg.max_agent_steps)

    async def stop(self) -> None:
        if self.tools:
            await self.tools.close()

    @property
    def web_allowed(self) -> bool:
        return bool(self.cfg.web and self.settings.safety.permissions.get("internet_read", False) and self.agent)

    def set_tools(self, tools: WebTools) -> None:
        """Swap the web back-end (tests use a simulated network)."""
        self.tools = tools
        self.agent = ResearchAgent(self.ctx.llm, tools, max_steps=self.cfg.max_agent_steps)

    async def query(self, q: dict) -> dict:
        self.stats["queries"] += 1
        topic = impersonal_question(str(q.get("question", "")))
        if topic is None:
            self.stats["refused_personal"] += 1
            return {"known": False, "answer": "", "confidence": 0.0, "source": "none",
                    "reason": "not a general-knowledge question"}
        key = topic.lower()
        if key in self._cache:
            return self._cache[key]
        fresh = time_sensitive(topic)
        book = None if fresh else await self._books(topic)
        if book and book["known"] and book["confidence"] >= 0.5:
            return book
        if self.web_allowed:
            found = await self._look_up(topic, fresh)
            if found:
                return found
        if fresh:  # no way to check: say what the books say, flagged as possibly out of date
            book = await self._books(topic)
            if book["known"]:
                book = {**book, "maybe_stale": True}
        book = book or {"known": False, "answer": "", "confidence": 0.0, "source": "none", "topic": topic}
        if not book["known"] and self.cfg.web and not self.web_allowed:
            book = {**book, "reason": "internet_off"}
        self.stats["known" if book["known"] else "unknown"] += 1
        return book

    async def recite(self, q: dict) -> dict:
        """One piece of the requested kind, from what it has read (a joke, a riddle, a poem, ...)."""
        kind = str(q.get("kind") or "joke").strip().lower()
        topic = str(q.get("topic") or "").strip()
        stock = STOCK.get(kind, [])
        res = await self.ctx.llm.complete(
            "recite", RECITE_SYSTEM, f"Kind: {kind}\nTopic: {topic or 'anything'}\nPiece:", schema=Piece,
            fallback=lambda: Piece(text=self.ctx.rng.choice(stock) if stock else ""), priority=2, max_tokens=120,
            temperature=0.9)
        text = (res.data.text or "").strip()
        if not text and stock:
            text = self.ctx.rng.choice(stock)
        self.recent = ([{"t": self.now(), "topic": f"{kind}{' about ' + topic if topic else ''}", "known": bool(text),
                         "answer": text, "via": "recited"}] + self.recent)[:12]
        return {"text": text, "kind": kind, "source": f"reading ({res.model})" if text else "none"}

    async def _look_up(self, topic: str, fresh: bool) -> dict | None:
        agent = self.agent if self.cfg.agent else ResearchAgent(_Symbolic(), self.tools, self.cfg.max_agent_steps)
        f = await agent.run(topic, time_sensitive=fresh)
        self.recent = ([{"t": self.now(), "topic": topic, "known": f.known, "answer": f.answer, "via": "web",
                         "method": f.method, "steps": f.steps, "sources": f.sources}] + self.recent)[:12]
        if not f.known:
            return None
        self.stats["looked_up"] += 1
        self.stats["known"] += 1
        names = list(dict.fromkeys(s.get("source") or "web" for s in f.sources))
        out = {"known": True, "answer": f.answer, "confidence": f.confidence, "kind": "web", "topic": topic,
               "source": "web (" + ", ".join(names) + ")", "sources": f.sources, "method": f.method,
               "retrieved": time.strftime("%Y-%m-%d")}
        if not fresh:
            self._cache[topic.lower()] = out
        await self.ask_one("memory.store", {
            "kind": "semantic", "content": f.answer, "source": out["source"], "importance": 0.35,
            "confidence": min(0.85, f.confidence),
            "context": {"question": topic, "sources": f.sources, "retrieved": out["retrieved"]}}, default=None)
        return out

    async def _books(self, topic: str) -> dict:
        res = await self.ctx.llm.complete(
            "knowledge_lookup", LIBRARY_SYSTEM, f"Question: {topic}?", schema=Lookup,
            fallback=lambda: Lookup(known=False), priority=2, max_tokens=140, temperature=0.2,
        )
        lk: Lookup = res.data
        answer = lk.answer.strip()
        known = lk.known and bool(answer) and not _PERSONAL.search(answer)  # a persona leaking in is discarded
        out = {"known": known, "answer": truncate(answer, 300) if known else "",
               "confidence": round(clamp(lk.confidence), 3) if known else 0.0, "kind": "reading",
               "source": f"reading ({res.model})" if known else "none", "topic": topic}
        if res.model != "rule" and known and not time_sensitive(topic):
            self._cache[topic.lower()] = out
        if known:
            await self.ask_one("memory.store", {
                "kind": "semantic", "content": out["answer"], "source": "reading",
                "importance": 0.3, "confidence": min(0.8, out["confidence"]),
                "context": {"question": topic}}, default=None)
        self.recent = ([{"t": self.now(), "topic": topic, "via": "books",
                         **{k: out[k] for k in ("known", "answer", "confidence")}}] + self.recent)[:12]
        return out

    def snapshot(self) -> dict:
        return {"stats": dict(self.stats), "recent": self.recent, "web_allowed": self.web_allowed,
                "web_search": bool(self.tools and self.tools.has_web_search),
                "web_stats": dict(self.tools.stats) if self.tools else {}}


class _Symbolic:
    """Stands in for the language model when the agent loop is switched off: the fixed pipeline runs."""
    is_symbolic = True
