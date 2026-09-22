"""Metacognition: monitoring the organism's own cognition from real internal metadata.

Represents uncertainty (confidence of attended contents), knowledge gaps (questions it
could not answer), conflicts (memory vs perception), source monitoring (was the last
answer from memory, perception, self-model or nothing?), model reliability (prediction
error statistics, language-model failures) and the current focus of attention.
Statements such as "I may be wrong" are produced only when the metadata supports them.
"""
from __future__ import annotations

from collections import deque

from ..core.events import (
    ErrorPayload, Event, EventType, MetaPayload, SpeechPayload, TickPayload, WorldConflictPayload, WorkspacePayload,
)
from ..core.module import CognitiveModule
from ..core.util import truncate

SOURCE_TEXT = {"memory": "memory", "perception": "direct perception", "self_model": "my self-model",
               "interoception": "my internal state", "world_model": "my world model", "working_memory": "working memory",
               "metacognition": "self-monitoring", "goals": "my goals", "language": "what I was just told"}


class Metacognition(CognitiveModule):
    name = "metacognition"
    subscriptions = (EventType.WORKSPACE_UPDATED, EventType.SPEECH_GENERATED, EventType.WORLD_CONFLICT,
                     EventType.MODULE_ERROR, EventType.TICK, EventType.UTTERANCE_UNDERSTOOD)

    def __init__(self, ctx):
        super().__init__(ctx)
        self.focus: str | None = None
        self.confidences: list[float] = []
        self.last_answer: dict | None = None
        self.last_question: str | None = None
        self.gaps: deque[str] = deque(maxlen=10)
        self.conflicts: deque[str] = deque(maxlen=10)
        self.failures: deque[str] = deque(maxlen=10)
        self.reports: deque[dict] = deque(maxlen=30)
        self._llm_failures_seen = 0
        self.pred_stats: dict = {}

    async def start(self) -> None:
        self.respond("metacog.assess", self.assess)

    def report(self, kind: str, content: str, conf: float, nominate: bool, evidence: list[str] | None = None) -> None:
        self.reports.appendleft({"t": self.now(), "kind": kind, "content": content})
        self.emit(EventType.METACOGNITIVE_REPORT, MetaPayload(kind=kind, content=content, confidence=conf,
                                                              evidence=evidence or []),
                  summary=f"(metacognition) {content}", nominated=nominate, salience=0.45 if nominate else 0.0,
                  confidence=conf)

    async def handle(self, event: Event) -> None:
        t = event.type
        if t == EventType.WORKSPACE_UPDATED:
            p = event.data(WorkspacePayload)
            if p.focus:
                self.focus = p.focus
            self.confidences = [float(i.event.get("confidence", 1.0)) for i in p.items]
        elif t == EventType.UTTERANCE_UNDERSTOOD:
            self.last_question = event.payload.get("analysis", {}).get("text")
        elif t == EventType.SPEECH_GENERATED:
            s = event.data(SpeechPayload)
            self.last_answer = {"text": s.text, "grounding": s.grounding, "gap": s.knowledge_gap}
            if s.knowledge_gap:
                topic = truncate(self.last_question or s.text, 80)
                self.gaps.appendleft(topic)
                self.report("knowledge_gap", f"I don't have enough information about: {topic}", 0.8, True,
                            [event.id])
            elif not s.grounding or s.grounding == ["none"]:
                self.report("source_monitoring", "My last utterance was not grounded in memory or perception; "
                                                  "it may be wrong.", 0.6, False, [event.id])
        elif t == EventType.WORLD_CONFLICT:
            c = event.data(WorldConflictPayload)
            text = (f"My memory conflicts with new information about {c.entity_id} ({c.attribute}): I believed "
                    f"'{c.believed}' ({c.believed_confidence:.2f}) but now have '{c.observed}' ({c.observed_confidence:.2f}).")
            self.conflicts.appendleft(text)
            self.report("conflict", text, 0.8, True, [event.id])
        elif t == EventType.MODULE_ERROR:
            e = event.data(ErrorPayload)
            self.failures.appendleft(f"{e.module}: {truncate(e.error, 100)}")
            self.report("module_failure", f"Part of my cognition failed ({e.module}).", 0.9, True, [event.id])
        elif t == EventType.TICK:
            tick = event.data(TickPayload)
            if tick.cycle % 30 == 0:
                self.pred_stats = await self.ask_one("prediction.stats", {}, default={}) or {}
                fails = self.ctx.llm.stats.failures
                if fails > self._llm_failures_seen:
                    self.report("performance", f"My language model failed {fails - self._llm_failures_seen} time(s) "
                                               "recently; I fell back to simpler symbolic processing.", 0.9, False)
                    self._llm_failures_seen = fails

    def overall_confidence(self) -> float | None:
        return round(sum(self.confidences) / len(self.confidences), 3) if self.confidences else None

    async def assess(self, _q: dict) -> dict:
        conf = self.overall_confidence()
        stmts: list[str] = []
        if self.focus:
            stmts.append(f"I am currently focusing on: {truncate(self.focus, 120)}.")
        if conf is not None:
            stmts.append(f"My confidence in what I am attending to is {conf:.2f}.")
            if conf < 0.6:
                stmts.append("I am uncertain, so I may be wrong.")
        if self.last_answer:
            src = [SOURCE_TEXT.get(g, g) for g in self.last_answer["grounding"] if g != "none"]
            if src:
                stmts.append(f"I generated my last answer from {' and '.join(src)}"
                             + (" rather than direct perception." if "direct perception" not in src else "."))
            else:
                stmts.append("My last answer had no source in memory or perception, so I may be wrong.")
        if self.gaps:
            stmts.append(f"I don't have enough information about: {self.gaps[0]}.")
        if self.conflicts:
            stmts.append(self.conflicts[0])
        errs = [v["mean_error"] for v in self.pred_stats.values() if v.get("n", 0) >= 3]
        if errs:
            m = sum(errs) / len(errs)
            stmts.append(f"My recent predictions had a mean error of {m:.2f}, so my models are "
                         f"{'fairly reliable' if m < 0.3 else 'not very reliable'}.")
        stmts.append(f"My language is produced by {self.ctx.llm.name}, which can make mistakes."
                     if not self.ctx.llm.is_symbolic else "My language is produced by simple symbolic rules.")
        return {"focus": self.focus, "overall_confidence": conf, "uncertainty": None if conf is None else round(1 - conf, 3),
                "last_answer": self.last_answer, "knowledge_gaps": list(self.gaps), "conflicts": list(self.conflicts),
                "failures": list(self.failures), "prediction_stats": self.pred_stats, "statements": stmts}

    def snapshot(self) -> dict:
        return {"focus": self.focus, "overall_confidence": self.overall_confidence(), "reports": list(self.reports)[:10],
                "knowledge_gaps": list(self.gaps)[:5], "conflicts": list(self.conflicts)[:5],
                "last_answer": self.last_answer}

    def trace_state(self) -> dict:
        return {"gaps": len(self.gaps), "conflicts": len(self.conflicts)}
