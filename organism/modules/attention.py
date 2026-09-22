"""Attention: decides what becomes globally available.

Any module can nominate an event (``event.nominated``). On every TICK the candidates compete.
Each candidate gets component scores (salience, novelty, urgency, goal relevance, emotional
relevance, uncertainty, prediction error) which are combined with a noisy-OR so that any strong
component can win on its own. Component weights are modulated by the organism's state:
curiosity amplifies novelty and surprise, fear/stress amplify urgency, low energy raises the
ignition threshold (conservation), sleep raises it a lot (gating).

Items above the (arousal-dependent) ignition threshold enter the workspace, subject to capacity.
Recency is not a factor: attention is *not* "the latest message".
"""
from __future__ import annotations

from collections import deque

import numpy as np

from ..core.events import (
    AttentionItem, AttentionPayload, EmotionPayload, Event, EventType, GoalPayload, Mode, TickPayload,
)
from ..core.module import CognitiveModule
from ..core.util import clamp, cosine

MODE_THRESHOLD_OFFSET = {Mode.AWAKE: 0.0, Mode.DROWSY: 0.1, Mode.ASLEEP: 0.45, Mode.DREAMING: 0.25}
INTERNAL_DREAM_TYPES = {EventType.DREAM_CONTENT, EventType.MEMORY_REPLAYED, EventType.SIMULATION_RESULT}
TYPE_BIAS = {
    EventType.UTTERANCE_UNDERSTOOD: 0.15,
    EventType.SYSTEM_BOOT: 0.3,
    EventType.SLEEP_ENDED: 0.2,
    EventType.WORLD_CONFLICT: 0.15,
    EventType.ACTION_REJECTED: 0.1,
}


class Attention(CognitiveModule):
    name = "attention"
    subscriptions = ("*",)

    def __init__(self, ctx):
        super().__init__(ctx)
        self.cfg = self.settings.attention
        self.pool: dict[str, tuple[Event, int]] = {}
        self.goals: dict[str, tuple[str, float, np.ndarray]] = {}
        self.emotion: dict[str, float] = {}
        self.body: dict = {"arousal": 0.5, "energy": 1.0, "mode": Mode.AWAKE}
        self.recent_broadcast: deque[tuple[float, np.ndarray, str]] = deque(maxlen=60)
        self.last: AttentionPayload | None = None
        self.focus_history: deque[dict] = deque(maxlen=50)

    async def start(self) -> None:
        self.respond("attention.focus", self._focus)

    async def _focus(self, _q: dict) -> dict | None:
        if not self.last:
            return None
        return self.last.model_dump(mode="json", exclude={"selected": {"__all__": {"event"}}})

    async def handle(self, event: Event) -> None:
        t = event.type
        if t == EventType.TICK:
            tick = event.data(TickPayload)
            self.body = tick.body.model_dump()
            self.compete(tick.cycle, tick.body.mode)
            return
        if t in (EventType.GOAL_CREATED, EventType.GOAL_UPDATED):
            g = event.data(GoalPayload).goal
            if g.status == "active":
                self.goals[g.id] = (g.description, g.priority, self.ctx.fast_embedder.embed_sync(g.description))
            else:
                self.goals.pop(g.id, None)
        elif t == EventType.EMOTION_CHANGED:
            self.emotion = event.data(EmotionPayload).state
        if event.nominated:
            self.pool[event.id] = (event, self.bus.cycle)

    # ------------------------------------------------------------------ scoring
    def components(self, ev: Event, vec: np.ndarray) -> dict[str, float]:
        now = self.now()
        sims = [cosine(vec, v) for ts, v, _ in self.recent_broadcast
                if now - ts < self.cfg.inhibition_of_return_s * 10]
        novelty = clamp(1.0 - max(sims, default=0.0))
        goal_rel = 0.0
        for _desc, prio, gvec in self.goals.values():
            goal_rel = max(goal_rel, clamp(cosine(vec, gvec) * 1.8) * prio)
        if ev.type in (EventType.GOAL_CREATED, EventType.GOAL_UPDATED):
            goal_rel = max(goal_rel, float(ev.payload.get("goal", {}).get("priority", 0.5)))
        return {
            "salience": clamp(ev.salience),
            # Novelty amplifies salient input rather than standing in for it: a faint novel
            # stimulus produces only a weak orienting response.
            "novelty": round(novelty * (0.4 + 0.6 * clamp(ev.salience)), 4),
            "urgency": clamp(ev.urgency),
            "goal_relevance": clamp(goal_rel),
            "emotional_relevance": clamp(abs(ev.emotional_value) + 0.5 * ev.importance),
            "uncertainty": clamp(ev.uncertainty or 0.0),
            "prediction_error": clamp(float(ev.payload.get("error", 0.0))) if ev.type == EventType.PREDICTION_ERROR else 0.0,
        }

    def modulated_weights(self) -> dict[str, float]:
        w = dict(self.cfg.weights)
        e = self.emotion
        curiosity = e.get("curiosity", 0.3)
        fear = e.get("fear", 0.0)
        w["novelty"] = clamp(w["novelty"] * (0.6 + curiosity + 0.5 * e.get("boredom", 0.0)))
        w["prediction_error"] = clamp(w["prediction_error"] * (0.7 + curiosity))
        w["urgency"] = clamp(w["urgency"] * (1.0 + fear + 0.5 * float(self.body.get("stress", 0.0))))
        w["emotional_relevance"] = clamp(w["emotional_relevance"] * (1.0 + 0.5 * fear))
        return w

    def score(self, comps: dict[str, float], weights: dict[str, float], bias: float) -> float:
        p_not = 1.0 - bias
        for k, c in comps.items():
            p_not *= 1.0 - clamp(weights.get(k, 0.0) * c)
        return clamp(1.0 - p_not)

    def threshold(self, mode: Mode) -> float:
        a = float(self.body.get("arousal", 0.5))
        energy = float(self.body.get("energy", 1.0))
        return clamp(self.cfg.ignition_threshold + MODE_THRESHOLD_OFFSET[mode]
                     - 0.15 * (a - 0.5) + 0.1 * (1.0 - energy), 0.05, 0.95)

    def compete(self, cycle: int, mode: Mode) -> AttentionPayload:
        ttl = self.cfg.candidate_ttl_cycles
        cands = [(e, c) for e, c in self.pool.values() if cycle - c <= ttl]
        weights = self.modulated_weights()
        theta = self.threshold(mode)
        scored: list[tuple[float, Event, dict, np.ndarray]] = []
        for ev, _c in cands:
            vec = self.ctx.fast_embedder.embed_sync(ev.summary or ev.type.value)
            comps = self.components(ev, vec)
            bias = TYPE_BIAS.get(ev.type, 0.0)
            if mode == Mode.DREAMING:
                bias += 0.35 if ev.type in INTERNAL_DREAM_TYPES else 0.0
            s = self.score(comps, weights, bias)
            scored.append((s, ev, comps, vec))
        scored.sort(key=lambda x: -x[0])
        capacity = self.settings.workspace.capacity
        winners = [x for x in scored if x[0] >= theta][:capacity]
        now = self.now()
        selected = []
        for s, ev, comps, vec in winners:
            self.recent_broadcast.append((now, vec, ev.summary))
            selected.append(AttentionItem(
                event_id=ev.id, event_type=ev.type, source=ev.source, summary=ev.summary,
                score=round(s, 4), components={k: round(v, 3) for k, v in comps.items()},
                event=ev.model_dump(mode="json"),
            ))
            self.pool.pop(ev.id, None)
        # Expire stale candidates (unattended information fades).
        for eid in [eid for eid, (_e, c) in self.pool.items() if cycle - c > ttl]:
            self.pool.pop(eid, None)
        payload = AttentionPayload(selected=selected, threshold=round(theta, 3), candidates=len(cands), mode=mode)
        self.last = payload
        if selected:
            self.focus_history.append({"cycle": cycle, "t": now, "focus": selected[0].summary,
                                       "score": selected[0].score})
        top = selected[0].summary if selected else "nothing"
        self.emit(EventType.ATTENTION_CHANGED, payload,
                  summary=f"attending to {top[:100]} ({len(selected)}/{len(cands)} ignited, θ={theta:.2f})")
        return payload

    # ------------------------------------------------------------------ observability
    def snapshot(self) -> dict:
        last = self.last
        return {
            "threshold": last.threshold if last else None,
            "candidates": last.candidates if last else 0,
            "pool": len(self.pool),
            "weights": {k: round(v, 3) for k, v in self.modulated_weights().items()},
            "selected": [{"summary": s.summary, "score": s.score, "components": s.components,
                          "type": s.event_type.value} for s in (last.selected if last else [])],
            "history": list(self.focus_history)[-15:],
        }

    def trace_state(self) -> dict:
        return {"pool": len(self.pool), "goals": len(self.goals)}

