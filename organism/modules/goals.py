"""Goals and motivation.

Persistent intrinsic goals (maintenance, understanding, social, learning) plus temporary goals
created from the situation (respond to a question, investigate a surprise, rest when tired).
Goal priorities are dynamic (e.g. rest priority grows with fatigue). Goals are broadcast so
they bias attention (goal relevance) and action selection.
"""
from __future__ import annotations

from ..core.events import (
    ActionPayload, ActionType, Event, EventType, Goal, GoalPayload, Mode, TickPayload, UtterancePayload,
    WorkspacePayload,
)
from ..core.module import CognitiveModule
from ..core.util import clamp, truncate

INTRINSIC_GOALS = [
    dict(id="g_maintain", description="Maintain my operational integrity: keep enough energy and rest when needed",
         priority=0.35, origin="survival", kind="maintenance", required_actions=["sleep"]),
    dict(id="g_social", description="Communicate helpfully and honestly with the people who talk to me",
         priority=0.7, origin="social", kind="primary", required_actions=["speak", "ask"]),
    dict(id="g_understand", description="Understand my environment, the people in it and unexpected events",
         priority=0.5, origin="intrinsic", kind="intrinsic", required_actions=["look", "investigate"]),
    dict(id="g_learn", description="Learn and remember important information",
         priority=0.45, origin="intrinsic", kind="secondary", required_actions=["remember"]),
]
RESPONSE_INTENTS = ("question", "command", "greeting", "farewell", "statement")


class Goals(CognitiveModule):
    name = "goals"
    subscriptions = (EventType.WORKSPACE_UPDATED, EventType.ACTION_EXECUTED, EventType.ACTION_REJECTED,
                     EventType.TICK, EventType.ACTION_APPROVED)

    def __init__(self, ctx):
        super().__init__(ctx)
        self.goals: dict[str, Goal] = {}

    async def start(self) -> None:
        for d in self.ctx.store.load_goals():
            g = Goal.model_validate(d)
            if g.status == "active" and g.kind == "temporary":
                g.status = "abandoned"  # situational goals do not survive a restart
                self._save(g)
            self.goals[g.id] = g
        for spec in INTRINSIC_GOALS:
            if spec["id"] not in self.goals:
                self.create(Goal(created_at=self.now(), **spec), announce=False)
        self.respond("goals.active", self._active)
        self.respond("executor.set_goal", lambda q: _true())

    async def _active(self, _q: dict) -> list[dict]:
        return [g.model_dump(mode="json") for g in self.active()]

    def active(self) -> list[Goal]:
        return sorted((g for g in self.goals.values() if g.status == "active"), key=lambda g: -g.priority)

    def _save(self, g: Goal) -> None:
        self.ctx.store.put_goal(g.id, g.status, g.model_dump(mode="json"), self.now())

    def create(self, g: Goal, announce: bool = True, salience: float = 0.4) -> Goal:
        self.goals[g.id] = g
        self._save(g)
        self.emit(EventType.GOAL_CREATED, GoalPayload(goal=g, change="created"),
                  summary=f"new goal ({g.origin}, p={g.priority:.2f}): {g.description}",
                  nominated=announce, salience=salience, importance=g.priority * 0.5)
        return g

    def update(self, g: Goal, change: str, **fields) -> None:
        for k, v in fields.items():
            setattr(g, k, v)
        self._save(g)
        self.emit(EventType.GOAL_UPDATED, GoalPayload(goal=g, change=change),
                  summary=f"goal {change}: {g.description} [{g.status}]",
                  nominated=g.status in ("achieved", "failed") and g.kind == "temporary", salience=0.3)

    async def handle(self, event: Event) -> None:
        now = self.now()
        if event.type == EventType.WORKSPACE_UPDATED:
            p = event.data(WorkspacePayload)
            for it in p.items:
                if it.event_id not in p.new_item_ids:
                    continue
                if it.event_type == EventType.UTTERANCE_UNDERSTOOD:
                    a = UtterancePayload.model_validate(it.event["payload"]).analysis
                    if a.addressed_to_self and a.intent in RESPONSE_INTENTS and not self._has_goal_for(it.event_id):
                        desc = (f"Carry out {a.speaker}'s request: \"{truncate(a.text, 60)}\"" if a.intent == "command"
                                else f"Respond to {a.speaker}: \"{truncate(a.text, 60)}\"")
                        self.create(Goal(description=desc, priority=0.85, origin="user", kind="temporary",
                                         created_at=now, deadline=now + 120, required_actions=["speak"],
                                         related_event=it.event_id, expected_reward=0.7), salience=0.35)
                elif it.event_type == EventType.PREDICTION_ERROR:
                    err = float(it.event.get("payload", {}).get("error", 0.0))
                    explorations = [g for g in self.active() if g.origin == "exploration"]
                    target = str(it.event.get("payload", {}).get("target", ""))
                    external = target in ("visual_scene", "conversation")
                    if err >= 0.5 and external and len(explorations) < 2:
                        self.create(Goal(description=f"Understand the unexpected: {truncate(it.summary, 80)}",
                                         priority=clamp(0.35 + 0.4 * err), origin="exploration", kind="temporary",
                                         created_at=now, deadline=now + 300, required_actions=["investigate"],
                                         related_event=it.event_id, expected_reward=0.4 * err))
        elif event.type == EventType.ACTION_EXECUTED:
            ap = event.data(ActionPayload)
            ok = ap.result.get("outcome") == "success"
            for g in self.active():
                related = (ap.spec.goal_id == g.id) or (ap.spec.about_event and ap.spec.about_event == g.related_event)
                silent = ap.spec.params.get("intention") == "stay_silent"
                if related and g.kind == "temporary" and ok and (ap.spec.action.value in g.required_actions or silent):
                    self.update(g, "achieved", status="achieved", progress=1.0)
                elif related and g.origin == "exploration" and ok and ap.spec.action in (
                        ActionType.INVESTIGATE, ActionType.LOOK):
                    self.update(g, "achieved", status="achieved", progress=1.0)
        elif event.type == EventType.ACTION_APPROVED:
            ap = event.data(ActionPayload)
            if ap.spec.action == ActionType.SET_GOAL:
                desc = str(ap.spec.params.get("description") or ap.spec.reason or "an unspecified goal")
                g = self.create(Goal(description=desc, priority=float(ap.spec.params.get("priority", 0.6)),
                                     origin="user" if ap.spec.params.get("from_user") else "task",
                                     kind="secondary", created_at=now, expected_reward=0.5))
                self.emit(EventType.ACTION_EXECUTED,
                          ActionPayload(decision_id=ap.decision_id, spec=ap.spec, status="executed",
                                        result={"outcome": "success", "goal_id": g.id}),
                          summary=f"Executed set_goal: {truncate(desc, 80)}")
        elif event.type == EventType.ACTION_REJECTED:
            ap = event.data(ActionPayload)
            for g in self.active():
                if ap.spec.goal_id == g.id and g.kind == "temporary":
                    self.update(g, "failed", status="failed")
        elif event.type == EventType.TICK:
            tick = event.data(TickPayload)
            for g in self.active():
                if g.deadline and now > g.deadline:
                    self.update(g, "expired", status="expired")
            maint = self.goals.get("g_maintain")
            if maint and tick.body.mode == Mode.AWAKE:
                p = round(clamp(0.25 + 0.7 * tick.body.fatigue), 2)
                if abs(p - maint.priority) >= 0.1:
                    self.update(maint, "reprioritised", priority=p)

    def _has_goal_for(self, event_id: str) -> bool:
        return any(g.related_event == event_id for g in self.goals.values())

    def snapshot(self) -> dict:
        recent_done = sorted((g for g in self.goals.values() if g.status != "active" and g.kind == "temporary"),
                             key=lambda g: -g.created_at)[:6]
        return {"active": [g.model_dump(mode="json") for g in self.active()],
                "recent": [{"description": g.description, "status": g.status} for g in recent_done]}

    def trace_state(self) -> dict:
        return {g.id: [g.status, round(g.priority, 2)] for g in self.active()}


async def _true() -> bool:
    return True
