"""Self-model: an explicit, persistent representation of the organism *as* an entity.

It keeps identity, body (sensors and the face it acts with), capabilities, limitations (static and learned
from failures), the current state (mode, arousal, valuation, focus, goal, thought, percept,
action), a timeline of its own activity ("five minutes ago I was ..."), temporal continuity
across restarts, and predictions about its own near future.

Self/world distinction: the world model holds external entities; the self-model holds the
observer. Percepts are represented *with attribution* ("I perceive via the camera: ...")
rather than as bare facts.
"""
from __future__ import annotations

import platform
from collections import deque

from ..core.events import (
    ActionPayload, ActionType, EmotionPayload, Event, EventType, GoalPayload, ModePayload, PerceptPayload,
    SelfStatePayload, SpeechPayload, ThoughtPayload, TickPayload, UtterancePayload, WorkspacePayload,
)
from ..core.module import CognitiveModule
from ..core.util import fmt_clock, fmt_datetime, humanize_duration, truncate

CAPABILITY_TEXT = {
    "speak": "speak (text output, optionally synthesised voice)", "ask": "ask questions",
    "look": "look (process camera images)", "remember": "deliberately store and recall memories",
    "investigate": "think through problems", "simulate": "imagine scenarios", "sleep": "sleep and consolidate memories",
    "wake": "wake up", "set_goal": "adopt new goals", "note": "write notes to a notebook (permission-controlled)",
    "wait": "stay silent and observe", "express": "make facial expressions",
}
MODE_TEXT = {"AWAKE": "awake", "DROWSY": "drowsy", "ASLEEP": "asleep", "DREAMING": "in a dream-like phase of sleep"}


class SelfModel(CognitiveModule):
    name = "self_model"
    subscriptions = (
        EventType.TICK, EventType.EMOTION_CHANGED, EventType.WORKSPACE_UPDATED, EventType.THOUGHT_GENERATED,
        EventType.GOAL_CREATED, EventType.GOAL_UPDATED, EventType.ACTION_EXECUTED, EventType.ACTION_REJECTED,
        EventType.SPEECH_GENERATED, EventType.PERCEPTION, EventType.MODE_CHANGED, EventType.UTTERANCE_UNDERSTOOD,
    )

    def __init__(self, ctx):
        super().__init__(ctx)
        ident = self.settings.identity
        self.identity: dict = {"name": ident.name, "version": ident.version, "description": ident.description,
                               "narrative": ""}
        self.current: dict = {"mode": "AWAKE", "arousal": None, "valence": None, "dominant_state": None,
                              "focus": None, "goal": None, "thought": None, "perception": None,
                              "last_action": None, "activity": "starting up", "interlocutor": None,
                              "location": self.settings.world.location_name}
        self.limitations: list[str] = []
        self.learned_limitations: list[str] = []
        self.timeline: deque[dict] = deque(maxlen=300)
        self.sensors: list[dict] = []
        self.face: dict | None = None
        self.body: dict = {}

    async def start(self) -> None:
        lc = self.ctx.store.kv_get("organism.lifecycle", {}) or {}
        saved = self.kv_load("state", {}) or {}
        self.identity.update({k: saved.get("identity", {}).get(k) for k in ("narrative",) if saved.get("identity")})
        if self.ctx.ltm_enabled:
            self.learned_limitations = saved.get("learned_limitations", [])
            self.timeline.extend(saved.get("timeline", [])[-100:])
        self.identity.update({"id": lc.get("organism_id"), "created_at": lc.get("created_at"),
                              "boot_count": lc.get("boot_count")})
        self.sensors = self._sensors()
        self.face = self._face()
        self.limitations = self._static_limitations()
        self.respond("self.describe", self.describe)
        self.respond("self.reflect", self.reflect)

    async def stop(self) -> None:
        self.persist()

    def persist(self) -> None:
        self.kv_save("state", {"identity": {"narrative": self.identity.get("narrative", "")},
                               "learned_limitations": self.learned_limitations if self.ctx.ltm_enabled else [],
                               "timeline": list(self.timeline)[-100:] if self.ctx.ltm_enabled else []})

    def _sensors(self) -> list[dict]:
        s = self.settings
        out = [{"id": "console", "modality": "text", "description": "a text channel (keyboard/dashboard)"}]
        if s.enabled("vision"):
            out.append({"id": "camera", "modality": "vision",
                        "description": "a camera" + (" (live)" if s.vision.enabled else " (images sent to me)")})
        if s.enabled("audition"):
            out.append({"id": "microphone", "modality": "audio",
                        "description": "a microphone" + (" (live)" if s.audio.microphone else " (recordings sent to me)")})
        out.append({"id": "interoception", "modality": "interoceptive",
                    "description": "internal signals: arousal, energy, fatigue, valuation"})
        return out

    def _face(self) -> dict | None:
        """The organism's body: an animated face it moves itself (when the expression module is on)."""
        if not self.settings.enabled("expression"):
            return None
        return {"description": "an animated face on a screen: two eyes, eyebrows and a mouth",
                "parts": {"eyes": "look left, right, up or down; close; wink", "eyebrows": "raise",
                          "mouth": "smile, grin, laugh, frown"},
                "note": "my eyes are drawn on the face; I see through the camera, and I look around by moving them"}

    def _static_limitations(self) -> list[str]:
        body = ("My body is only a face: I have no arms or legs, so I cannot walk, move around or touch anything"
                if self.face else "I have no physical body, so I cannot move or touch anything")
        lim = [body,
               f"I only perceive through: {', '.join(x['id'] for x in self.sensors if x['id'] != 'interoception')}",
               "I perceive nothing while I am switched off",
               "My internal states are functional variables; I cannot verify that they are experiences"]
        if self.ctx.llm.is_symbolic:
            lim.append("I am running without a neural language model, so my language is simple and rule-based")
        else:
            lim.append(f"My general knowledge comes from reading ({self.ctx.llm.name}), which can be wrong")
        return lim

    def capabilities(self) -> list[str]:
        acts = [t.split(".", 1)[1] for t in self.bus.topics() if t.startswith("executor.")]
        return [CAPABILITY_TEXT.get(a, a) for a in sorted(acts)]

    # ------------------------------------------------------------------ tracking
    def _activity(self, text: str) -> None:
        self.current["activity"] = text
        if not self.timeline or self.timeline[-1]["activity"] != text:
            self.timeline.append({"t": self.now(), "activity": text})

    async def handle(self, event: Event) -> None:
        t = event.type
        c = self.current
        if t == EventType.TICK:
            tick = event.data(TickPayload)
            self.body = tick.body.model_dump(mode="json")
            c["mode"], c["arousal"] = tick.body.mode.value, round(tick.body.arousal, 3)
            if tick.cycle % 25 == 0:
                self.persist()
        elif t == EventType.EMOTION_CHANGED:
            p = event.data(EmotionPayload)
            c["valence"] = round(p.valence, 3)
            c["dominant_state"] = max(p.state, key=lambda k: p.state[k] - (0.35 if k == "curiosity" else 0.1))
        elif t == EventType.WORKSPACE_UPDATED:
            p = event.data(WorkspacePayload)
            if p.focus and p.new_item_ids:
                c["focus"] = p.focus
        elif t == EventType.THOUGHT_GENERATED:
            c["thought"] = event.data(ThoughtPayload).content
            if event.payload.get("kind") != "decision":
                self._activity("thinking: " + truncate(c["thought"], 80))
        elif t in (EventType.GOAL_CREATED, EventType.GOAL_UPDATED):
            g = event.data(GoalPayload).goal
            if g.status == "active" and (not c["goal"] or g.priority >= c["goal"]["priority"]):
                c["goal"] = {"description": g.description, "priority": g.priority}
            elif c["goal"] and c["goal"]["description"] == g.description and g.status != "active":
                c["goal"] = None
        elif t == EventType.ACTION_EXECUTED:
            p = event.data(ActionPayload)
            if p.spec.action != ActionType.WAIT:
                c["last_action"] = f"{p.spec.action.value}: {truncate(p.spec.reason, 80)}"
                self._activity(f"{p.spec.action.value}ing" if p.spec.action.value != "speak" else "talking")
        elif t == EventType.ACTION_REJECTED:
            p = event.data(ActionPayload)
            lim = p.reason
            if lim and lim not in self.learned_limitations and "rate limit" not in lim:
                self.learned_limitations.append(lim)
                self.emit(EventType.SELF_STATE_CHANGED,
                          SelfStatePayload(changes={"new_limitation": lim}, reason="an action was blocked"),
                          summary=f"I learned a limitation: {lim}", nominated=True, salience=0.4)
        elif t == EventType.SPEECH_GENERATED:
            s = event.data(SpeechPayload)
            c["last_action"] = f'said "{truncate(s.text, 80)}"'
            self._activity(f"talking with {s.addressed_to}")
        elif t == EventType.PERCEPTION:
            p = event.data(PerceptPayload)
            if not p.self_generated:
                c["perception"] = {"sensor": p.sensor, "modality": p.modality.value, "content": truncate(p.description, 160),
                                   "confidence": round(event.confidence, 2), "t": event.timestamp}
        elif t == EventType.UTTERANCE_UNDERSTOOD:
            a = event.data(UtterancePayload).analysis
            c["interlocutor"] = a.speaker
            self._activity(f"listening to {a.speaker}")
        elif t == EventType.MODE_CHANGED:
            p = event.data(ModePayload)
            self._activity(MODE_TEXT.get(p.current.value, p.current.value.lower()))

    # ------------------------------------------------------------------ queries
    def previous_activity(self, seconds_ago: float) -> dict | None:
        cutoff = self.now() - seconds_ago
        prev = [e for e in self.timeline if e["t"] <= cutoff]
        return prev[-1] if prev else None

    def temporal(self) -> dict:
        tmp = self.ctx.temporal
        now = self.now()
        out = {"clock": fmt_datetime(now), "awake_for": humanize_duration(now - tmp.boot_time) if tmp.boot_time else None,
               "downtime": humanize_duration(tmp.downtime) if tmp.downtime else None,
               "episode": tmp.episode_id, "sleep_cycles": tmp.sleep_cycles}
        five = self.previous_activity(300)
        if five:
            out["five_minutes_ago"] = f"{five['activity']} (at {fmt_clock(five['t'])})"
        return out

    async def describe(self, q: dict) -> dict:
        c = self.current
        ident = dict(self.identity)
        percept = c.get("perception")
        perspective = [f"I am {ident['name']} {ident['version']}.",
                       f"I am currently {MODE_TEXT.get(c['mode'], c['mode'])}."]
        if self.face:
            perspective.append(f"My body is {self.face['description']}.")
        if percept:
            age = self.now() - percept["t"]
            perspective.append(f"I perceived via my {percept['sensor']} ({percept['modality']}, confidence "
                               f"{percept['confidence']}, {humanize_duration(age)} ago): {percept['content']}")
        if c.get("focus"):
            perspective.append(f"My attention is on: {truncate(c['focus'], 120)}")
        if c.get("goal"):
            perspective.append(f"My current goal: {c['goal']['description']}")
        if c.get("thought"):
            perspective.append(f"My latest thought: {truncate(c['thought'], 120)}")
        perspective.append("The things I perceive belong to the world; my thoughts, goals, memories and actions are mine.")
        brief = {"identity": ident, "current": {**c}, "perspective": perspective, "face": self.face}
        if q.get("brief"):
            return brief
        future = []
        b = self.body
        if b and b.get("mode") == "AWAKE":
            remaining = (1.0 - float(b.get("sleep_pressure", 0))) * self.settings.brainstem.wake_period_s
            future.append(f"I expect to need sleep in roughly {humanize_duration(remaining)}.")
        return {**brief, "temporal": self.temporal(), "capabilities": self.capabilities(),
                "limitations": self.limitations + self.learned_limitations, "sensors": self.sensors,
                "body": b, "predicted_future": future,
                "self_world_boundary": {"self": ["my sensors", "my thoughts", "my memories", "my goals", "my actions",
                                                 "my internal state"],
                                        "world": "entities in my world model (people, objects, places, events)"}}

    async def reflect(self, _q: dict) -> str:
        auto = await self.ask_one("memory.recent", {"kinds": ["autobiographical"], "n": 8}, default=[]) or []
        epi = await self.ask_one("memory.recent", {"kinds": ["episodic"], "n": 6}, default=[]) or []
        name = self.identity["name"]

        def rule() -> str:
            n = self.identity.get("boot_count") or 1
            body = " My body is a face that I can move." if self.face else ""
            text = f"I am {name}, an experimental artificial cognitive system.{body} I have been started {n} times."
            if auto:
                text += f" Recently, {truncate(auto[0]['content'], 120)}"
            if epi:
                text += f" Among my experiences: {truncate(epi[0]['content'], 100)}"
            return text

        text = rule()  # the organism's own account of itself, not a language model's
        self.identity["narrative"] = text
        self.persist()
        self.emit(EventType.SELF_STATE_CHANGED,
                  SelfStatePayload(changes={"narrative": text}, reason="self-reflection during sleep"),
                  summary=f"updated my self-narrative: {truncate(text, 100)}")
        return text

    def snapshot(self) -> dict:
        return {"identity": self.identity, "current": self.current, "temporal": self.temporal(),
                "capabilities": self.capabilities(), "limitations": self.limitations + self.learned_limitations,
                "sensors": self.sensors, "face": self.face, "timeline": list(self.timeline)[-12:],
                "platform": f"{platform.system()} {platform.machine()}"}

    def trace_state(self) -> dict:
        c = self.current
        return {"mode": c["mode"], "activity": c["activity"], "goal": (c["goal"] or {}).get("description")}
