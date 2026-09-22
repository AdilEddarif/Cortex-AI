"""Working memory: a small, decaying, capacity-limited store of the active context.

* slots      - attended items (percepts, thoughts, memories, goals) with activation that decays
               exponentially and is refreshed when the item is re-broadcast (rehearsal).
* dialog     - the current conversation (phonological-loop-like buffer), turns expire.
* questions  - unresolved questions addressed to the organism.
Eviction removes the item with the lowest activation x importance.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass

from ..core.events import (
    Event, EventType, GoalPayload, SpeechPayload, UtterancePayload, WMPayload, WorkspacePayload,
)
from ..core.module import CognitiveModule
from ..core.util import cosine, half_life_decay

KIND_BY_TYPE = {
    EventType.UTTERANCE_UNDERSTOOD: "utterance", EventType.PERCEPTION: "percept",
    EventType.THOUGHT_GENERATED: "thought", EventType.MEMORY_RETRIEVED: "memory",
    EventType.GOAL_CREATED: "goal", EventType.PREDICTION_ERROR: "surprise",
    EventType.SPEECH_GENERATED: "speech", EventType.DREAM_CONTENT: "dream",
    EventType.SIMULATION_RESULT: "imagination", EventType.BODY_STATE: "body",
}


@dataclass
class WMItem:
    event_id: str
    content: str
    kind: str
    source: str
    modality: str
    activation: float
    importance: float
    created: float
    refreshed: float


class WorkingMemory(CognitiveModule):
    name = "working_memory"
    subscriptions = (
        EventType.WORKSPACE_UPDATED, EventType.TICK, EventType.SPEECH_GENERATED,
        EventType.UTTERANCE_UNDERSTOOD, EventType.GOAL_CREATED, EventType.GOAL_UPDATED,
    )

    def __init__(self, ctx):
        super().__init__(ctx)
        self.cfg = self.settings.working_memory
        self.slots: list[WMItem] = []
        self.dialog: deque[dict] = deque(maxlen=self.cfg.dialog_turns)
        self.questions: dict[str, dict] = {}
        self.current_goal: dict | None = None
        self._last_decay = self.now()

    async def start(self) -> None:
        self.respond("wm.contents", self._contents)

    async def _contents(self, q: dict) -> dict:
        items = list(self.slots)
        query = q.get("query")
        if query:
            qv = self.ctx.fast_embedder.embed_sync(query)
            items.sort(key=lambda i: -(i.activation * (0.3 + cosine(qv, self.ctx.fast_embedder.embed_sync(i.content)))))
        else:
            items.sort(key=lambda i: -i.activation)
        now = self.now()
        return {
            "items": [asdict(i) for i in items],
            "dialog": [d for d in self.dialog if now - d["t"] <= self.cfg.dialog_ttl_s],
            "unresolved_questions": list(self.questions.values()),
            "current_goal": self.current_goal,
        }

    async def handle(self, event: Event) -> None:
        t = event.type
        if t == EventType.WORKSPACE_UPDATED:
            p = event.data(WorkspacePayload)
            new_ids = set(p.new_item_ids)
            evicted = []
            for it in p.items:
                existing = next((s for s in self.slots if s.event_id == it.event_id), None)
                if existing:
                    existing.activation = max(existing.activation, it.score)
                    existing.refreshed = self.now()
                elif it.event_id in new_ids:
                    evicted += self.insert(WMItem(
                        event_id=it.event_id, content=it.summary, kind=KIND_BY_TYPE.get(it.event_type, "item"),
                        source=it.source, modality=it.modality.value, activation=max(it.score, 0.3),
                        importance=float(it.event.get("importance", 0.0)) + float(it.event.get("urgency", 0.0)),
                        created=self.now(), refreshed=self.now()))
            if new_ids or evicted:
                self.emit(EventType.WM_UPDATED, WMPayload(items=[asdict(s) for s in self.slots], evicted=evicted),
                          summary=f"working memory: {len(self.slots)}/{self.cfg.capacity} items")
        elif t == EventType.UTTERANCE_UNDERSTOOD:
            a = event.data(UtterancePayload).analysis
            self.dialog.append({"speaker": a.speaker, "text": a.text, "t": event.timestamp, "event_id": event.id})
            if a.addressed_to_self and a.intent in ("question", "command"):
                self.questions[event.id] = {"event_id": event.id, "text": a.text, "speaker": a.speaker,
                                            "t": event.timestamp}
        elif t == EventType.SPEECH_GENERATED:
            s = event.data(SpeechPayload)
            self.dialog.append({"speaker": "self", "text": s.text, "t": event.timestamp, "event_id": event.id})
            if s.in_reply_to:
                self.questions.pop(s.in_reply_to, None)
        elif t in (EventType.GOAL_CREATED, EventType.GOAL_UPDATED):
            g = event.data(GoalPayload).goal
            if g.status == "active" and (self.current_goal is None or g.priority >= self.current_goal["priority"]):
                self.current_goal = {"id": g.id, "description": g.description, "priority": g.priority}
            elif self.current_goal and g.id == self.current_goal["id"] and g.status != "active":
                self.current_goal = None
        elif t == EventType.TICK:
            self.decay()

    def insert(self, item: WMItem) -> list[str]:
        self.slots.append(item)
        evicted = []
        while len(self.slots) > self.cfg.capacity:
            victim = min(self.slots, key=lambda s: s.activation * (0.5 + s.importance))
            self.slots.remove(victim)
            evicted.append(victim.content)
        return evicted

    def decay(self) -> None:
        now = self.now()
        f = half_life_decay(now - self._last_decay, self.cfg.half_life_s)
        self._last_decay = now
        for s in self.slots:
            s.activation *= f
        self.slots = [s for s in self.slots if s.activation >= 0.05]
        for qid in [k for k, v in self.questions.items() if now - v["t"] > self.cfg.dialog_ttl_s]:
            self.questions.pop(qid)

    def snapshot(self) -> dict:
        now = self.now()
        return {
            "capacity": self.cfg.capacity,
            "items": [{"content": s.content, "kind": s.kind, "activation": round(s.activation, 3)}
                      for s in sorted(self.slots, key=lambda s: -s.activation)],
            "dialog": [{"speaker": d["speaker"], "text": d["text"]} for d in self.dialog
                       if now - d["t"] <= self.cfg.dialog_ttl_s],
            "unresolved_questions": [q["text"] for q in self.questions.values()],
            "current_goal": self.current_goal,
        }

    def trace_state(self) -> dict:
        return {"slots": [s.event_id for s in self.slots], "q": len(self.questions)}
