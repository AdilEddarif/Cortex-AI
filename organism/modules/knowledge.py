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
"""
from __future__ import annotations

import re

from pydantic import BaseModel, Field

from ..core.module import CognitiveModule
from ..core.util import clamp, content_words, truncate

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
_SITUATIONAL = re.compile(r"\b(here|now|today|tonight|right now|this room|in front)\b", re.I)


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
        self._cache: dict[str, dict] = {}
        self.recent: list[dict] = []
        self.stats = {"queries": 0, "refused_personal": 0, "known": 0, "unknown": 0}

    async def start(self) -> None:
        self.respond("knowledge.query", self.query)

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
        res = await self.ctx.llm.complete(
            "knowledge_lookup", LIBRARY_SYSTEM, f"Question: {topic}?", schema=Lookup,
            fallback=lambda: Lookup(known=False), priority=2, max_tokens=140, temperature=0.2,
        )
        lk: Lookup = res.data
        answer = lk.answer.strip()
        known = lk.known and bool(answer) and not _PERSONAL.search(answer)  # a persona leaking in is discarded
        out = {"known": known, "answer": truncate(answer, 300) if known else "",
               "confidence": round(clamp(lk.confidence), 3) if known else 0.0,
               "source": f"reading ({res.model})" if known else "none", "topic": topic}
        self.stats["known" if known else "unknown"] += 1
        if res.model != "rule":
            self._cache[key] = out
        if known:
            await self.ask_one("memory.store", {
                "kind": "semantic", "content": out["answer"], "source": "reading",
                "importance": 0.3, "confidence": min(0.8, out["confidence"]),
                "context": {"question": topic}}, default=None)
        self.recent = ([{"t": self.now(), "topic": topic, **{k: out[k] for k in ("known", "answer", "confidence")}}]
                       + self.recent)[:12]
        return out

    def snapshot(self) -> dict:
        return {"stats": dict(self.stats), "recent": self.recent}
