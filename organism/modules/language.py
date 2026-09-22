"""Language system (Wernicke/Broca-like roles). One subsystem, not the whole brain.

Understanding: linguistic percepts (typed text, speech transcripts) are parsed into a
structured ``LanguageAnalysis`` and nominated for attention as UTTERANCE_UNDERSTOOD.

Generation: ``language.generate`` turns a communicative intention into words. It gathers the
organism's *actual* state through the bus (self-model, perception/world model, memories,
working memory, interoception, metacognition, goals) and composes the reply from that state
with its own rules. The language model never speaks for the organism: for general-knowledge
questions it is consulted through ``knowledge.query`` (like remembering something read in a
book) and the answer is woven into the organism's own sentence. Verbal reports about internal
state are logged as VERBAL_REPORT next to the snapshot they describe; they never modify the state.
"""
from __future__ import annotations

import asyncio

from ..core.events import (
    Event, EventType, LanguageAnalysis, Modality, PerceptPayload, UtterancePayload, VerbalReportPayload,
)
from ..core.module import CognitiveModule
from ..core.util import truncate
from .language_rules import compose_reply, parse_utterance

class Language(CognitiveModule):
    name = "language"
    subscriptions = (EventType.PERCEPTION,)

    def __init__(self, ctx):
        super().__init__(ctx)
        self.last_generation: dict | None = None
        self.parsed = 0

    async def start(self) -> None:
        self.respond("language.generate", self.generate)
        self.respond("language.parse", self._parse)

    async def _parse(self, q: dict) -> dict:
        return parse_utterance(q.get("text", ""), q.get("speaker", "user"),
                               self.settings.identity.name, q.get("channel", "console")).model_dump()

    # ------------------------------------------------------------------ understanding
    async def handle(self, event: Event) -> None:
        p = event.data(PerceptPayload)
        if p.self_generated:
            return
        if p.text is not None:
            text, speaker, channel = p.text.text, p.text.speaker, p.text.channel
        elif p.audio is not None and p.audio.kind == "speech" and p.audio.transcript:
            text, speaker, channel = p.audio.transcript, p.audio.speaker or "someone", "microphone"
        else:
            return
        a = parse_utterance(text, speaker, self.settings.identity.name, channel)
        self.parsed += 1
        urgent = a.addressed_to_self and a.intent in ("question", "command", "greeting")
        self.emit(
            EventType.UTTERANCE_UNDERSTOOD,
            UtterancePayload(analysis=a, percept_id=event.id, modality=p.modality),
            summary=f'{speaker} said: "{truncate(text, 140)}"', modality=p.modality,
            confidence=event.confidence, nominated=True,
            salience=round(max(event.salience, 0.75 if a.addressed_to_self else 0.4), 3),
            urgency=0.7 if urgent else 0.3, emotional_value=a.sentiment,
        )

    # ------------------------------------------------------------------ generation
    async def gather(self, a: LanguageAnalysis) -> dict:
        topics = {
            "self": ("self.describe", {}),
            "body": ("body.state", {}),
            "emotion": ("emotion.state", {}),
            "world": ("world.describe", {}),
            "memories": ("memory.recall", {"query": a.text, "k": 5}),
            "facts": ("memory.facts", {}),
            "recent_episodes": ("memory.recent", {"kinds": ["episodic"], "n": 120}),
            "wm": ("wm.contents", {"query": a.text}),
            "meta": ("metacog.assess", {"topic": a.text}),
            "goals": ("goals.active", {}),
            "user": ("world.query", {"id": f"person:{a.speaker.lower()}"}),
        }
        if a.asks_about == "dream":
            topics["dreams"] = ("memory.recent", {"kinds": ["dream"], "n": 3})
        if a.asks_about == "change":
            topics["surprises"] = ("prediction.recent_errors", {"max_age_s": 300})
        if a.asks_about == "past":
            topics["timeline"] = ("memory.episodes", {"text": a.text})
        if a.asks_about in ("thinking", "reason"):
            topics["thoughts"] = ("thought.recent", {})
        if a.intent == "question" and a.asks_about == "general":  # something one might have read about
            topics["knowledge"] = ("knowledge.query", {"question": a.text})
        keys = list(topics)
        results = await asyncio.gather(*(self.ask_one(t, q, timeout=20.0) for t, q in topics.values()))
        ctx = {k: v for k, v in zip(keys, results)}
        for k in ("memories", "facts", "recent_episodes", "goals", "dreams"):
            ctx[k] = ctx.get(k) or []
        ctx["user_name"] = self._user_name(a, ctx)
        return ctx

    @staticmethod
    def _user_name(a: LanguageAnalysis, ctx: dict) -> str | None:
        for f in a.facts:  # just told
            if f.relation == "name":
                return f.object
        user = ctx.get("user") or {}
        name = (user.get("attributes") or {}).get("name")
        if name and name.get("confidence", 0) >= 0.3:
            return name["value"]
        for f in ctx.get("facts", []):
            if f.get("relation") == "name" and str(f.get("subject", "")).lower() in (a.speaker.lower(), "user"):
                return f.get("object")
        return None

    async def generate(self, q: dict) -> dict:
        a = LanguageAnalysis.model_validate(q["analysis"]) if q.get("analysis") else LanguageAnalysis(
            text=q.get("content", ""), intent="statement")
        intention = q.get("intention", "answer")
        ctx = await self.gather(a)
        ctx["last_thought"] = q.get("thought")
        draft = compose_reply(intention, a, ctx)
        if intention == "stay_silent":
            return {**draft, "text": ""}
        text = draft["text"]
        model = (ctx.get("knowledge") or {}).get("source", "rule") if "knowledge" in draft["grounding"] else "rule"
        out = {"text": text, "grounding": draft["grounding"], "knowledge_gap": draft["knowledge_gap"],
               "model": model, "draft": draft["text"]}
        if draft.get("report_topic"):
            self.emit(EventType.VERBAL_REPORT,
                      VerbalReportPayload(topic=draft["report_topic"], report=text,
                                          state_snapshot=draft.get("report_snapshot") or {}),
                      summary=f"verbal report ({draft['report_topic']}): {truncate(text, 120)}",
                      modality=Modality.INTERNAL)
        self.last_generation = {"intention": intention, "heard": a.text, "text": text, "model": model,
                                "grounding": draft["grounding"]}
        return out

    def snapshot(self) -> dict:
        return {"parsed": self.parsed, "last_generation": self.last_generation}
