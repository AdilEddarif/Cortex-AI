"""Imagination: offline simulation of possible futures and dream-like recombination.

``imagination.simulate``  predicts the outcome, value and risks of an action *without acting*,
                          using the world model, learned action statistics (procedural memory)
                          and the current valuation state. Optionally refined by a language model.
``imagination.dream``     recombines memory fragments into an internally coherent dream-like
                          scenario (used during DREAMING).
executor ``simulate``     deliberate imagining requested by a person or by thought.
All products are tagged ``imagined`` so they are never confused with perception or memory.
"""
from __future__ import annotations

import re

from pydantic import BaseModel, Field

from ..core.events import (
    ActionPayload, ActionSpec, ActionType, EmotionPayload, Event, EventType, Modality, SimulationPayload, TickPayload,
)
from ..core.module import CognitiveModule
from ..core.util import clamp, content_words, truncate
from .temporal_memory import noun_heads

class SimOutcome(BaseModel):
    predicted_outcome: str
    success_probability: float = Field(0.5, ge=0, le=1)
    value: float = Field(0.0, ge=-1, le=1)
    risks: list[str] = Field(default_factory=list)


class DreamText(BaseModel):
    narrative: str


class Imagination(CognitiveModule):
    name = "imagination"
    subscriptions = (EventType.ACTION_APPROVED, EventType.EMOTION_CHANGED, EventType.TICK, EventType.ACTION_EXECUTED)

    def __init__(self, ctx):
        super().__init__(ctx)
        self.cfg = self.settings.imagination
        self.emotion: dict = {}
        self.body: dict = {}
        self.stats: dict[str, list[int]] = {}
        self.recent: list[dict] = []

    async def start(self) -> None:
        self.respond("imagination.simulate", self.simulate)
        self.respond("imagination.dream", self.dream)
        self.respond("executor.simulate", _true)

    async def handle(self, event: Event) -> None:
        if event.type == EventType.EMOTION_CHANGED:
            self.emotion = event.data(EmotionPayload).state
        elif event.type == EventType.TICK:
            self.body = event.data(TickPayload).body.model_dump(mode="json")
        elif event.type == EventType.ACTION_EXECUTED:
            p = event.data(ActionPayload)
            s = self.stats.setdefault(p.spec.action.value, [0, 0])
            s[0] += 1
            s[1] += int(p.result.get("outcome") == "success")
        elif event.type == EventType.ACTION_APPROVED:
            p = event.data(ActionPayload)
            if p.spec.action == ActionType.SIMULATE:
                await self._imagine_on_request(p)

    # ------------------------------------------------------------------ action rollouts
    def _success_p(self, action: str, prior: float) -> float:
        n, ok = self.stats.get(action, [0, 0])
        return (ok + 2 * prior) / (n + 2)  # Beta-prior smoothing of learned success rates

    def rule_simulate(self, spec: ActionSpec) -> SimOutcome:
        a = spec.action
        e = self.emotion
        fatigue = float(self.body.get("fatigue", 0.0))
        addressed = bool(spec.about_event)
        p = self._success_p(a.value, spec.confidence)
        risks: list[str] = []
        if spec.params.get("from_user") and a != ActionType.MOVE:
            # Doing what someone asked me to do is itself the social goal being served.
            return SimOutcome(predicted_outcome=f"I do what I was asked ({a.value}) and the person sees it",
                              success_probability=round(p, 3), value=round(0.6 * p, 3), risks=[])
        if a == ActionType.SPEAK:
            intention = spec.params.get("intention", "answer")
            value = 0.6 if addressed else 0.1
            if intention == "answer":
                value += 0.1
            outcome = "the person hears my reply; the conversation continues and my goal to respond is met"
            if e.get("fear", 0) > 0.5:
                risks.append("I might say something wrong while under threat")
        elif a == ActionType.WAIT:
            value = -0.4 if addressed and spec.params.get("intention") != "stay_silent" else 0.05
            outcome = "the person may feel ignored" if value < 0 else "I keep observing; nothing changes"
        elif a == ActionType.ASK:
            value = 0.3 + 0.3 * e.get("curiosity", 0.35)
            outcome = "I receive information that reduces my uncertainty"
        elif a == ActionType.SLEEP:
            value = fatigue - 0.3 - (0.4 if addressed else 0.0)
            outcome = "I restore energy and consolidate memories, but I stop responding for a while"
            if addressed:
                risks.append("I would leave a conversation unanswered")
        elif a == ActionType.LOOK:
            value = 0.3 + 0.3 * float(self.body.get("novelty", 0.0))
            outcome = "I refresh my visual model of the surroundings"
        elif a == ActionType.INVESTIGATE:
            value = 0.2 + 0.3 * e.get("curiosity", 0.35)
            outcome = "I understand the unexpected event better"
        elif a == ActionType.MOVE:
            value, p = -0.5, 0.0
            outcome = "nothing: I have no body"
            risks.append("the action cannot be executed")
        else:
            value = 0.2
            outcome = spec.expected_outcome or "a modest change"
        return SimOutcome(predicted_outcome=outcome, success_probability=round(p, 3),
                          value=round(clamp(value * p + (value if value < 0 else 0) * (1 - p), -1, 1), 3), risks=risks)

    async def simulate(self, q: dict) -> dict:
        spec = ActionSpec.model_validate(q["spec"])
        out = self.rule_simulate(spec)
        self.recent = ([{"t": self.now(), "action": spec.action.value, "outcome": out.predicted_outcome,
                         "value": out.value, "p": out.success_probability}] + self.recent)[:20]
        self.emit(EventType.SIMULATION_RESULT,
                  SimulationPayload(scenario=f"if I {spec.action.value}", action=spec,
                                    predicted_outcome=out.predicted_outcome, success_probability=out.success_probability,
                                    value=out.value, risks=out.risks),
                  summary=f"imagined: if I {spec.action.value} -> {truncate(out.predicted_outcome, 90)} (v={out.value:.2f})",
                  modality=Modality.IMAGINATION, confidence=out.success_probability)
        return out.model_dump()

    # ------------------------------------------------------------------ dreams & imagining
    def rule_dream(self, fragments: list[str]) -> DreamText:
        if not fragments:
            return DreamText(narrative="I drift through an empty, quiet space with no memories to hold on to.")
        cleaned = [truncate(re.sub(r"\buser\b", "you", f.replace('said to me:', 'said')).rstrip('.'), 90)
                   for f in fragments]
        heads = [w for f in fragments for w in noun_heads(f.split(":", 1)[-1]) if len(w) > 3]
        words = sorted(set(heads) or {w for f in fragments for w in content_words(f)
                                      if w not in ("said", "user", "thought", "heard", "acknowledge")},
                       key=lambda w: (-len(w), w))
        twist = f", and somehow everything turns into {self.ctx.rng.choice(words[:4])}" if words else ""
        if len(cleaned) == 1:
            return DreamText(narrative=f"In a dream-like simulation, {cleaned[0]}, but everything is subtly different{twist}.")
        return DreamText(narrative=(f"In a dream-like simulation, {cleaned[0]}; then, without transition, "
                                    f"{cleaned[1]}{twist}."))

    async def dream(self, q: dict) -> dict:
        frags = [f for f in q.get("fragments", []) if f]
        return {"narrative": self.rule_dream(frags).narrative, "model": "rule"}

    async def _imagine_on_request(self, p: ActionPayload) -> None:
        scenario = str(p.spec.params.get("scenario") or p.spec.reason)
        mems = await self.ask_one("memory.recall", {"query": scenario, "k": 2}, default=[])
        frags = [scenario] + [m["content"] for m in mems or []]
        narrative = f"I imagine {scenario}."
        if len(frags) > 1:
            narrative += f" It reminds me of: {truncate(frags[1], 80)}."
        book = await self.ask_one("knowledge.query", {"question": f"what is {scenario}"}, default=None) or {}
        if book.get("known"):  # what it has read colours the picture; the picturing is its own
            narrative += f" I remember reading that {truncate(book['answer'], 160).rstrip('.')}."
        self.emit(EventType.SIMULATION_RESULT,
                  SimulationPayload(scenario=scenario, predicted_outcome=truncate(narrative, 200), narrative=narrative,
                                    value=0.2, success_probability=1.0),
                  summary=f"(imagined) {truncate(narrative, 140)}", modality=Modality.IMAGINATION,
                  nominated=True, salience=0.5, confidence=0.3)
        self.emit(EventType.ACTION_EXECUTED,
                  ActionPayload(decision_id=p.decision_id, spec=p.spec, status="executed",
                                result={"outcome": "success", "narrative": narrative}),
                  summary=f"Executed simulate: {truncate(narrative, 80)}", modality=Modality.MOTOR)

    def snapshot(self) -> dict:
        return {"recent": self.recent[:10], "learned_success": {k: v for k, v in self.stats.items()}}

    def trace_state(self) -> dict:
        return {"stats": self.stats}


async def _true(_q: dict) -> bool:
    return True
