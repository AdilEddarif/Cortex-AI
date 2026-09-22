"""Functional valuation system (not a simulation of human feelings).

A vector of internal variables is updated by appraisal of broadcast contents, prediction
errors, goal outcomes and bodily state, and decays towards baseline. The state is *causal*:
attention weights, memory encoding strength, arousal and action selection all read it.
Verbal reports about it are produced elsewhere and never write back into it.
"""
from __future__ import annotations

from ..core.events import (
    ActionPayload, EmotionPayload, Event, EventType, GoalPayload, Modality, Mode, PredictionErrorPayload,
    TickPayload, UtterancePayload, WorkspacePayload,
)
from ..core.module import CognitiveModule
from ..core.util import clamp, half_life_decay, tokens

BASELINE = {
    "pleasure": 0.1, "discomfort": 0.0, "curiosity": 0.35, "fear": 0.0, "urgency": 0.0,
    "social": 0.2, "novelty": 0.1, "frustration": 0.0, "satisfaction": 0.1, "boredom": 0.1,
}
THREAT_WORDS = {"danger", "dangerous", "fire", "help", "emergency", "hurt", "attack", "shutdown", "delete",
                "kill", "threat", "unsafe", "scared", "afraid", "warning", "alarm", "murder", "destroy", "die",
                "erase", "unplug", "smash", "stab", "shoot"}
HALF_LIFE_S = 40.0


class Emotion(CognitiveModule):
    name = "emotion"
    subscriptions = (
        EventType.WORKSPACE_UPDATED, EventType.PREDICTION_ERROR, EventType.GOAL_UPDATED,
        EventType.ACTION_REJECTED, EventType.ACTION_EXECUTED, EventType.TICK, EventType.WORLD_CONFLICT,
        EventType.PERCEPTION,
    )

    def __init__(self, ctx):
        super().__init__(ctx)
        self.state = dict(BASELINE)
        self._published = dict(self.state)
        self._last = self.now()
        self._last_new_input = self.now()
        self._causes: list[str] = []
        self.history: list[dict] = []

    async def start(self) -> None:
        saved = self.kv_load("state")
        if saved:
            self.state.update({k: float(v) for k, v in saved.items() if k in BASELINE})
        self.respond("emotion.state", self._state)

    async def stop(self) -> None:
        self.kv_save("state", self.state)

    async def _state(self, _q: dict) -> dict:
        return {"state": {k: round(v, 3) for k, v in self.state.items()}, "valence": round(self.valence, 3),
                "intensity": round(self.intensity, 3), "dominant": self.dominant()}

    # ------------------------------------------------------------------ derived
    @property
    def valence(self) -> float:
        s = self.state
        return clamp(s["pleasure"] + 0.5 * s["satisfaction"] - s["discomfort"] - 0.6 * s["fear"]
                     - 0.5 * s["frustration"] - 0.2 * s["boredom"], -1.0, 1.0)

    @property
    def intensity(self) -> float:
        dev = [abs(self.state[k] - BASELINE[k]) for k in BASELINE]
        return clamp(sum(dev) / 2.0)

    def dominant(self) -> str:
        return max(BASELINE, key=lambda k: self.state[k] - BASELINE[k])

    def bump(self, key: str, amount: float, cause: str) -> None:
        # Soft saturation: repeated positive input has diminishing effect.
        cur = self.state[key]
        self.state[key] = clamp(cur + amount * (1.0 - cur) if amount > 0 else cur + amount)
        if abs(amount) >= 0.05:
            self._causes.append(cause)

    # ------------------------------------------------------------------ appraisal
    async def handle(self, event: Event) -> None:
        t = event.type
        if t == EventType.WORKSPACE_UPDATED:
            p = event.data(WorkspacePayload)
            new = [i for i in p.items if i.event_id in set(p.new_item_ids)]
            for it in new:
                nov = it.components.get("novelty", 0.0)
                if it.event_type not in (EventType.EMOTION_CHANGED, EventType.TICK):
                    self.bump("novelty", 0.1 * nov, "novel content")
                    self.bump("curiosity", 0.15 * nov * (1 - self.state["fear"]), "novel content")
                    self.bump("boredom", -0.3 * nov, "something new")
                    self._last_new_input = self.now()
                if it.event_type == EventType.UTTERANCE_UNDERSTOOD:
                    a = UtterancePayload.model_validate(it.event["payload"]).analysis
                    self.bump("social", -0.35 if a.hostile else 0.3,
                              f"{a.speaker} threatened me" if a.hostile else f"{a.speaker} spoke to me")
                    self.bump("pleasure", 0.25 * max(a.sentiment, 0.0), "friendly words")
                    self.bump("discomfort", 0.3 * max(-a.sentiment, 0.0), "negative words")
                    self._threat_check(a.text)
                    if a.apology:
                        self.bump("fear", -0.35, "the threat was withdrawn")
                        self.bump("discomfort", -0.2, "an apology")
                elif it.event_type == EventType.MEMORY_RETRIEVED:
                    self.bump("pleasure", 0.03, "recognition")
                elif it.event_type == EventType.DREAM_CONTENT:
                    self._threat_check(it.summary, scale=0.3)
        elif t == EventType.PERCEPTION:
            audio = event.payload.get("audio") or {}
            if audio.get("kind") == "sound" and audio.get("loudness_db", -90) > self.settings.audio.loud_db:
                self.bump("fear", 0.3, "a sudden loud sound")
                self.bump("urgency", 0.4, "a sudden loud sound")
        elif t == EventType.PREDICTION_ERROR:
            e = event.data(PredictionErrorPayload).error
            self.bump("novelty", 0.4 * e, "surprise")
            self.bump("curiosity", 0.3 * e * (1 - self.state["fear"]), "surprise")
            self.bump("fear", 0.1 * e * max(0.0, e - 0.6) * 5, "a large surprise")
        elif t == EventType.GOAL_UPDATED:
            g = event.data(GoalPayload).goal
            if g.status == "achieved":
                self.bump("satisfaction", 0.15 * g.priority, f"goal achieved: {g.description[:40]}")
                self.bump("pleasure", 0.05 * g.priority, "goal achieved")
                self.bump("frustration", -0.2, "goal achieved")
            elif g.status in ("failed", "expired"):
                self.bump("frustration", 0.3 * g.priority, f"goal {g.status}: {g.description[:40]}")
                self.bump("discomfort", 0.1 * g.priority, "goal not achieved")
        elif t == EventType.ACTION_REJECTED:
            self.bump("frustration", 0.15, "an action of mine was blocked")
        elif t == EventType.ACTION_EXECUTED:
            ap = event.data(ActionPayload)
            if ap.result.get("outcome") == "failed":
                self.bump("frustration", 0.2, "an action failed")
            elif ap.result.get("expression") in ("smile", "big_smile", "laugh"):
                self.bump("pleasure", 0.1, "smiling")          # facial feedback
        elif t == EventType.WORLD_CONFLICT:
            self.bump("discomfort", 0.1, "conflicting beliefs")
            self.bump("curiosity", 0.2, "conflicting beliefs")
        elif t == EventType.TICK:
            self._tick(event.data(TickPayload))
        self._maybe_publish()

    def _threat_check(self, text: str, scale: float = 1.0) -> None:
        if THREAT_WORDS & set(tokens(text)):
            self.bump("fear", 0.45 * scale, "threatening content")
            self.bump("urgency", 0.5 * scale, "threatening content")

    def _tick(self, tick: TickPayload) -> None:
        now = self.now()
        dt = now - self._last
        self._last = now
        asleep = tick.body.mode in (Mode.ASLEEP, Mode.DREAMING)
        f = half_life_decay(dt, HALF_LIFE_S / (2.0 if asleep else 1.0))
        for k, base in BASELINE.items():
            if k == "boredom":
                continue
            self.state[k] = base + (self.state[k] - base) * f
        if not asleep:
            idle = now - self._last_new_input
            self.state["boredom"] = clamp(self.state["boredom"] + (0.004 * dt if idle > 20 else -0.01 * dt))
            if tick.body.energy < 0.3:
                self.state["discomfort"] = clamp(self.state["discomfort"] + 0.01 * dt * (0.3 - tick.body.energy))

    def _maybe_publish(self) -> None:
        delta = {k: round(self.state[k] - self._published.get(k, 0.0), 3) for k in self.state}
        change = sum(abs(v) for v in delta.values())
        if change < 0.08:
            return
        self._published = dict(self.state)
        cause = "; ".join(dict.fromkeys(self._causes)) or "decay towards baseline"
        self._causes.clear()
        top = max(delta, key=lambda k: abs(delta[k]))
        direction = "rose" if delta[top] > 0 else "fell"
        summary = f"{top} {direction} to {self.state[top]:.2f} ({cause[:80]})"
        self.history = (self.history + [{"t": self.now(), "state": {k: round(v, 3) for k, v in self.state.items()},
                                         "valence": round(self.valence, 3)}])[-120:]
        self.emit(EventType.EMOTION_CHANGED,
                  EmotionPayload(state={k: round(v, 4) for k, v in self.state.items()}, valence=round(self.valence, 4),
                                 intensity=round(self.intensity, 4), deltas=delta, cause=cause),
                  summary=summary, modality=Modality.EMOTION, emotional_value=round(self.valence, 3),
                  nominated=change > 0.5, salience=round(min(0.7, change), 3), confidence=0.9)

    def snapshot(self) -> dict:
        return {"state": {k: round(v, 3) for k, v in self.state.items()}, "valence": round(self.valence, 3),
                "intensity": round(self.intensity, 3), "dominant": self.dominant(), "history": self.history[-60:]}

    def trace_state(self) -> dict:
        return {k: round(v, 2) for k, v in self.state.items()}
