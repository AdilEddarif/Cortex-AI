"""Comprehension: turning free speech into a structured request (a Wernicke-like language faculty).

The rule parser (``language_rules``) recognises the forms it knows. Anything else — a typo, an
unusual phrasing, a request nobody wrote a pattern for — lands here, and the language model is used
for the one job it is genuinely good at and that costs the architecture nothing: **translation**.
It returns a small, fixed structure (what is wanted, of what kind, about what, which capability) and
never an answer, a fact, an opinion or a word that is spoken. Everything else still happens in the
cortex's own modules, and without a language model the rules simply stand on their own.

It also holds the cortex's **learned skills**: when the person explains what they mean ("when I say
X, do Y"), the pairing is remembered and matched (by wording and by meaning) from then on, so the
same request is understood next time without the model.
"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from ..core.module import CognitiveModule
from ..core.util import clamp, cosine, truncate

COMPREHEND_SYSTEM = """You are the language-understanding faculty of a cognitive system. You translate what
someone said into a structured request. You never answer it, never add facts and never speak.
Fields:
- wants: "perform" (they want a piece produced: a joke, story, riddle, poem, song, quote),
  "act" (they want the system to do something it can do), "ask_fact" (a question about the world),
  "ask_self" (a question about the system itself), "social" (greeting, thanks, small talk),
  "unclear" (you cannot tell).
- what: the thing asked for, one or two words ("joke", "poem", "smile", "sleep")
- topic: what it should be about, if they said ("rain"), else empty
- action: exactly one of [{capabilities}] when wants="act" and it clearly matches one, else empty
- confidence: 0-1
Judge only what was said. If they ask *about* something ("what is a joke?"), that is ask_fact, not perform."""

PRODUCIBLE = {"joke", "story", "riddle", "poem", "song", "quote", "fun fact", "tongue twister", "limerick"}
# Ways of saying "from now on, when I say X, do Y".
TEACH = re.compile(r"^(?:please\s+)?(?:from now on,?\s+)?(?:when|whenever|if)\s+i\s+(?:say|ask for|type|write)\s+"
                   r"[\"']?(?P<trigger>.+?)[\"']?\s*(?:,\s*|\s+then\s+|\s*:\s*)(?:you\s+should\s+|please\s+|"
                   r"then\s+)?(?P<action>.+)$", re.I)


class Comprehension(BaseModel):
    wants: Literal["perform", "act", "ask_fact", "ask_self", "social", "unclear"] = "unclear"
    what: str = ""
    topic: str = ""
    action: str = ""
    confidence: float = Field(0.5, ge=0, le=1)


class Understanding(CognitiveModule):
    name = "comprehension"

    def __init__(self, ctx):
        super().__init__(ctx)
        self.skills: list[dict] = []          # {trigger, means, vector, taught_at, uses}
        self.recent: list[dict] = []
        self.stats = {"parsed": 0, "by_model": 0, "by_skill": 0, "unclear": 0, "taught": 0}

    async def start(self) -> None:
        self.respond("comprehend.parse", self.parse)
        self.respond("skills.learn", self.learn)
        self.respond("skills.list", self._list)
        self.skills = [dict(s) for s in (self.kv_load("skills", []) or [])]
        for s in self.skills:
            s["vector"] = self.ctx.fast_embedder.embed_sync(s["trigger"])

    async def stop(self) -> None:
        self.kv_save("skills", [{k: v for k, v in s.items() if k != "vector"} for s in self.skills])

    # ------------------------------------------------------------------ learned skills
    async def learn(self, q: dict) -> dict:
        """Remember that one wording means one thing ("when I say hop, close your eyes")."""
        trigger, means = str(q.get("trigger", "")).strip(" .!?"), str(q.get("means", "")).strip(" .!?")
        if not trigger or not means:
            return {"learned": False}
        self.skills = [s for s in self.skills if s["trigger"].lower() != trigger.lower()]
        self.skills.append({"trigger": trigger, "means": means, "taught_at": self.now(), "uses": 0,
                            "vector": self.ctx.fast_embedder.embed_sync(trigger)})
        self.skills = self.skills[-50:]
        self.stats["taught"] += 1
        await self.ask_one("memory.store", {
            "kind": "procedural", "content": f'When they say "{trigger}", I should {means}.', "importance": 0.6,
            "source": "taught", "confidence": 0.9, "context": {"trigger": trigger, "means": means}}, default=None)
        return {"learned": True, "trigger": trigger, "means": means}

    async def _list(self, _q: dict) -> list[dict]:
        return [{k: s[k] for k in ("trigger", "means", "uses")} for s in self.skills]

    def match_skill(self, text: str) -> dict | None:
        """A learned wording, matched exactly or by meaning."""
        low = text.strip(" .!?").lower()
        for s in self.skills:
            if s["trigger"].lower() == low:
                return s
        if not self.skills:
            return None
        vec = self.ctx.fast_embedder.embed_sync(text)
        best = max(self.skills, key=lambda s: cosine(vec, s["vector"]))
        return best if cosine(vec, best["vector"]) >= 0.86 else None

    # ------------------------------------------------------------------ understanding
    async def parse(self, q: dict) -> dict:
        """A patch for the rule parser's analysis: what this utterance is really asking for."""
        text = str(q.get("text", ""))
        self.stats["parsed"] += 1
        skill = self.match_skill(text)
        if skill:
            skill["uses"] += 1
            self.stats["by_skill"] += 1
            self._note(text, {"via": "learned skill", "means": skill["means"]})
            return {"learned": True, "text": skill["means"], "trigger": skill["trigger"]}
        if self.ctx.llm.is_symbolic:
            return {}
        res = await self.ctx.llm.complete(
            "comprehend", COMPREHEND_SYSTEM.format(capabilities=", ".join(sorted(self.capabilities()))),
            f'Someone said: "{truncate(text, 300)}"\nStructured request:', schema=Comprehension,
            fallback=lambda: Comprehension(), priority=2, max_tokens=120, temperature=0.1)
        c: Comprehension = res.data
        if res.model == "rule" or c.wants == "unclear" or c.confidence < 0.4:
            self.stats["unclear"] += 1
            self._note(text, {"via": "model", "wants": c.wants, "confidence": c.confidence})
            return {}
        self.stats["by_model"] += 1
        patch = self._to_patch(c)
        self._note(text, {"via": "model", **patch})
        return patch

    def _to_patch(self, c: Comprehension) -> dict:
        what = c.what.strip().lower().removeprefix("a ").removeprefix("an ").rstrip("s")
        conf = round(clamp(c.confidence), 3)
        if c.wants == "perform":
            kind = what if what in PRODUCIBLE else next((k for k in PRODUCIBLE if k in what), "")
            if kind:
                arg = f"{kind} about {c.topic.strip()}" if c.topic.strip() else kind
                return {"intent": "command", "command": "recite", "command_arg": arg, "confidence": conf}
            return {"intent": "command", "request": what or c.topic, "unsupported": True, "confidence": conf}
        if c.wants == "act":
            action = c.action.strip().lower()
            if action and action in self.capabilities():
                return {"intent": "command", "command": action, "command_arg": c.topic or what, "confidence": conf}
            return {"intent": "command", "request": what or c.topic, "unsupported": True, "confidence": conf}
        if c.wants == "ask_fact":
            return {"intent": "question", "is_question": True, "asks_about": "general", "confidence": conf}
        if c.wants == "ask_self":
            return {"intent": "question", "is_question": True, "asks_about": "capability" if "do" in what else None,
                    "confidence": conf}
        return {}

    def capabilities(self) -> set[str]:
        return {t.split(".", 1)[1] for t in self.bus.topics() if t.startswith("executor.")} | {"recite"}

    def _note(self, text: str, info: dict) -> None:
        self.recent = ([{"t": self.now(), "text": truncate(text, 80), **info}] + self.recent)[:12]

    def snapshot(self) -> dict:
        return {"stats": dict(self.stats), "recent": self.recent,
                "skills": [{k: s[k] for k in ("trigger", "means", "uses")} for s in self.skills]}
