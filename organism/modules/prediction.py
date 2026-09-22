"""Predictive processing.

The organism continuously predicts:
  * the visual scene (persistence prior: what I saw is still there),
  * conversational events (after I ask a question, a reply should come; the speaker's tone
    should stay close to its running average; nobody should suddenly talk after a long silence),
  * the outcome of its own actions (success probability from the selection confidence),
  * its own next internal state (arousal trend).
Mismatches produce PREDICTION_ERROR events (nominated in proportion to the error), which drive
attention, curiosity, memory encoding and exploration goals. Running error statistics estimate how
reliable the organism's own models are (used by metacognition).
"""
from __future__ import annotations

from collections import deque

from ..core.events import (
    ActionPayload, Event, EventType, Modality, Mode, PerceptPayload, PredictionErrorPayload, PredictionPayload,
    SpeechPayload, TickPayload, UtterancePayload,
)
from ..core.module import CognitiveModule
from ..core.util import clamp, jaccard_distance, new_id, truncate


class Prediction(CognitiveModule):
    name = "prediction"
    subscriptions = (
        EventType.PERCEPTION, EventType.UTTERANCE_UNDERSTOOD, EventType.SPEECH_GENERATED, EventType.ACTION_SELECTED,
        EventType.ACTION_EXECUTED, EventType.ACTION_REJECTED, EventType.TICK,
    )

    def __init__(self, ctx):
        super().__init__(ctx)
        self.scene: set[str] | None = None
        self.expect_reply: dict | None = None
        self.sentiment: dict[str, float] = {}
        self.last_utterance_t: float | None = None
        self.action_expectations: dict[str, dict] = {}
        self.arousal_hist: deque[float] = deque(maxlen=5)
        self.predicted_arousal: float | None = None
        self.errors: deque[dict] = deque(maxlen=40)
        self.predictions: deque[dict] = deque(maxlen=40)
        self.stats: dict[str, dict] = {}

    async def start(self) -> None:
        self.respond("prediction.stats", self._stats)

    async def _stats(self, _q: dict) -> dict:
        return {k: {"n": v["n"], "mean_error": round(v["sum"] / max(1, v["n"]), 3)} for k, v in self.stats.items()}

    # ------------------------------------------------------------------ helpers
    def predict(self, target: str, predicted, confidence: float, horizon: float | None = None) -> str:
        pid = new_id("p_")
        exp = self.now() + horizon if horizon else None
        self.predictions.appendleft({"t": self.now(), "target": target, "predicted": predicted,
                                     "confidence": round(confidence, 2)})
        self.emit(EventType.PREDICTION_MADE,
                  PredictionPayload(prediction_id=pid, target=target, predicted=predicted, confidence=confidence,
                                    expires_at=exp),
                  summary=f"predict {target}: {truncate(str(predicted), 100)}", confidence=confidence)
        return pid

    def error(self, pid: str, target: str, predicted, observed, err: float, summary: str,
              modality: Modality = Modality.INTERNAL, report_threshold: float = 0.25) -> None:
        err = round(clamp(err), 3)
        st = self.stats.setdefault(target, {"n": 0, "sum": 0.0})
        st["n"] += 1
        st["sum"] += err
        if err < report_threshold:
            return
        self.errors.appendleft({"t": self.now(), "target": target, "error": err, "summary": summary})
        self.emit(EventType.PREDICTION_ERROR,
                  PredictionErrorPayload(prediction_id=pid, target=target, predicted=predicted, observed=observed, error=err),
                  summary=summary, modality=modality, nominated=err >= 0.35, salience=err, importance=0.5 * err,
                  confidence=0.8)

    # ------------------------------------------------------------------ events
    async def handle(self, event: Event) -> None:
        t = event.type
        now = self.now()
        if t == EventType.PERCEPTION:
            p = event.data(PerceptPayload)
            if p.self_generated or p.vision is None:
                return
            labels = {o.label for o in p.vision.objects}
            if self.scene is not None:
                err = jaccard_distance(labels, self.scene)
                added, missing = sorted(labels - self.scene), sorted(self.scene - labels)
                desc = []
                if added:
                    desc.append(f"new: {', '.join(added)}")
                if missing:
                    desc.append(f"gone: {', '.join(missing)}")
                self.error("scene", "visual_scene", sorted(self.scene), sorted(labels), err,
                           f"The visual scene changed unexpectedly ({'; '.join(desc) or 'different'})", Modality.VISION)
            self.scene = labels
            self.predict("visual_scene", sorted(labels), 0.7)
        elif t == EventType.UTTERANCE_UNDERSTOOD:
            a = event.data(UtterancePayload).analysis
            if self.expect_reply and a.speaker == self.expect_reply["from"]:
                self.stats.setdefault("conversation", {"n": 0, "sum": 0.0})["n"] += 1
                self.expect_reply = None
            elif self.last_utterance_t is None or now - self.last_utterance_t > 600:
                self.error(new_id("p_"), "conversation", "silence", a.text, 0.35,
                           f"Unexpected speech after a long silence from {a.speaker}", Modality.AUDIO)
            mean = self.sentiment.get(a.speaker)
            if mean is not None:
                diff = abs(a.sentiment - mean) / 2.0
                if diff >= 0.3:
                    tone = "more negative" if a.sentiment < mean else "more positive"
                    self.error(new_id("p_"), "speaker_tone", round(mean, 2), a.sentiment, diff,
                               f"{a.speaker}'s tone became {tone} than I expected", Modality.TEXT)
                self.sentiment[a.speaker] = 0.7 * mean + 0.3 * a.sentiment
            else:
                self.sentiment[a.speaker] = a.sentiment
            self.last_utterance_t = now
        elif t == EventType.SPEECH_GENERATED:
            s = event.data(SpeechPayload)
            if s.text.strip().endswith("?"):
                pid = self.predict("conversation", f"{s.addressed_to} will reply within 90s", 0.7, horizon=90)
                self.expect_reply = {"pid": pid, "from": s.addressed_to, "deadline": now + 90, "question": s.text}
        elif t == EventType.ACTION_SELECTED:
            p = event.data(ActionPayload)
            self.action_expectations[p.decision_id] = {"p": p.spec.confidence, "action": p.spec.action.value,
                                                       "expected": p.spec.expected_outcome}
        elif t in (EventType.ACTION_EXECUTED, EventType.ACTION_REJECTED):
            p = event.data(ActionPayload)
            exp = self.action_expectations.pop(p.decision_id, None)
            if exp:
                ok = t == EventType.ACTION_EXECUTED and p.result.get("outcome") == "success"
                self.error(p.decision_id, f"action:{exp['action']}", exp["expected"],
                           p.result.get("outcome", "rejected"), abs((1.0 if ok else 0.0) - exp["p"]),
                           f"My {exp['action']} action {'succeeded' if ok else 'did not work'} "
                           f"(I expected p={exp['p']:.2f} of success)", Modality.MOTOR, report_threshold=0.4)
        elif t == EventType.TICK:
            tick = event.data(TickPayload)
            a = tick.body.arousal
            if self.predicted_arousal is not None and tick.body.mode == Mode.AWAKE:
                self.error("arousal", "internal_state", round(self.predicted_arousal, 3), a,
                           abs(a - self.predicted_arousal) * 1.5,
                           f"My arousal changed unexpectedly ({self.predicted_arousal:.2f} -> {a:.2f})",
                           Modality.INTEROCEPTIVE, report_threshold=0.5)
            self.arousal_hist.append(a)
            h = list(self.arousal_hist)
            trend = (h[-1] - h[0]) / max(1, len(h) - 1) if len(h) > 1 else 0.0
            self.predicted_arousal = clamp(a + 0.5 * trend)
            if self.expect_reply and now > self.expect_reply["deadline"]:
                exp, self.expect_reply = self.expect_reply, None
                self.error(exp["pid"], "conversation", "a reply", "no reply", 0.5,
                           f"I expected {exp['from']} to answer my question, but no reply came", Modality.AUDIO)

    def snapshot(self) -> dict:
        return {"recent_predictions": list(self.predictions)[:10], "recent_errors": list(self.errors)[:10],
                "accuracy": {k: round(1 - v["sum"] / max(1, v["n"]), 3) for k, v in self.stats.items()},
                "expecting_reply": bool(self.expect_reply), "scene": sorted(self.scene or [])}

    def trace_state(self) -> dict:
        return {"scene": sorted(self.scene or []), "expect": bool(self.expect_reply)}
