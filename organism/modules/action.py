"""Action selection (basal-ganglia-like) and planning.

Every cycle, from the broadcast workspace contents, the module:
  1. generates candidate actions (responding to people, thought proposals, goal-driven
     actions, homeostatic actions, curiosity-driven actions, and always ``wait``);
  2. evaluates them: base value + goal support + emotional modulation + learned habit
     strength - energetic cost;
  3. for important/close decisions, imagines each option (``imagination.simulate``) and
     folds the predicted value and risk into the score;
  4. selects at most one action per effector channel (vocal / other).
A request made of several steps becomes a committed *plan*: its steps run one after another,
each waiting for the previous one to finish (and to be seen or heard), each through safety.
Selections go to the safety layer (ACTION_SELECTED -> ACTION_APPROVED | ACTION_REJECTED);
only approved actions are executed by effector modules. Silence is a first-class choice:
when a person speaks but a reply is not warranted, a deliberate ``wait`` is selected and logged.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from ..core.events import (
    VOCAL_ACTIONS, ActionPayload, ActionSpec, ActionType, EmotionPayload, Event, EventType, GoalPayload, Modality,
    Mode, TickPayload, UtterancePayload, WorkspaceItem, WorkspacePayload,
)
from ..core.module import CognitiveModule
from ..core.util import clamp, new_id, truncate
from .language_rules import count_text
from .thought import rule_deliberate

ENERGY_COST = {
    ActionType.SPEAK: 0.08, ActionType.ASK: 0.08, ActionType.WAIT: 0.0, ActionType.LOOK: 0.05,
    ActionType.REMEMBER: 0.03, ActionType.INVESTIGATE: 0.1, ActionType.SIMULATE: 0.12, ActionType.SLEEP: 0.0,
    ActionType.WAKE: 0.0, ActionType.SET_GOAL: 0.02, ActionType.NOTE: 0.05, ActionType.MOVE: 0.2,
    ActionType.EXPRESS: 0.01, ActionType.REVOKE: 0.0,
}
EXPECTED = {
    ActionType.SPEAK: "the listener receives my reply and the conversation continues",
    ActionType.ASK: "I obtain missing information",
    ActionType.WAIT: "nothing changes; I keep observing",
    ActionType.LOOK: "my visual model of the surroundings is refreshed",
    ActionType.REMEMBER: "the information is stored in / retrieved from long-term memory",
    ActionType.INVESTIGATE: "I understand the situation better",
    ActionType.SIMULATE: "I obtain an imagined scenario",
    ActionType.SLEEP: "energy is restored and memories are consolidated",
    ActionType.SET_GOAL: "a new goal guides my behaviour",
    ActionType.NOTE: "a note is written to my notebook",
    ActionType.MOVE: "my body moves",
    ActionType.EXPRESS: "my face shows the expression",
    ActionType.REVOKE: "I no longer have that permission",
}
COMMAND_ACTIONS = {
    "sleep": ActionType.SLEEP, "wake": ActionType.WAKE, "look": ActionType.LOOK, "remember": ActionType.REMEMBER,
    "simulate": ActionType.SIMULATE, "note": ActionType.NOTE, "move": ActionType.MOVE, "set_goal": ActionType.SET_GOAL,
    "express": ActionType.EXPRESS,
}


@dataclass
class Candidate:
    spec: ActionSpec
    base: float
    utility: float = 0.0
    notes: list[str] = field(default_factory=list)


class Action(CognitiveModule):
    name = "action"
    subscriptions = (
        EventType.WORKSPACE_UPDATED, EventType.ACTION_PROPOSED, EventType.ACTION_APPROVED, EventType.ACTION_EXECUTED,
        EventType.ACTION_REJECTED, EventType.EMOTION_CHANGED, EventType.TICK, EventType.GOAL_CREATED,
        EventType.GOAL_UPDATED,
    )

    def __init__(self, ctx):
        super().__init__(ctx)
        self.body: dict = {"arousal": 0.5, "energy": 1.0, "fatigue": 0.0, "mode": Mode.AWAKE}
        self.emotion: dict = {}
        self.goals: dict[str, dict] = {}
        self.proposals: list[tuple[float, ActionSpec]] = []
        self.pending: dict[str, tuple[ActionSpec, float]] = {}
        self.responded: set[str] = set()
        self.habits: dict[str, float] = {}
        self.last_social = -1e9
        self.last_noticed = -1e9
        self.decisions: list[dict] = []
        self.plan: dict | None = None  # {"steps": deque[ActionSpec], "waiting": decision id, "next_at": t}
        self.notebook = Path(self.settings.data_dir) / "notebook.md"

    async def start(self) -> None:
        self.habits = self.kv_load("habits", {}) or {}
        self.respond("executor.wait", _true)
        self.respond("executor.note", _true)
        self.respond("action.recent", self._recent)

    async def stop(self) -> None:
        self.kv_save("habits", self.habits)

    async def _recent(self, _q: dict) -> list[dict]:
        return self.decisions[:10]

    # ------------------------------------------------------------------ events
    async def handle(self, event: Event) -> None:
        t = event.type
        if t == EventType.TICK:
            self.body = event.data(TickPayload).body.model_dump()
            now = self.now()
            for did in [d for d, (_s, ts) in self.pending.items() if now - ts > 60]:
                self.pending.pop(did)
            self._advance_plan(now)
        elif t == EventType.EMOTION_CHANGED:
            self.emotion = event.data(EmotionPayload).state
        elif t in (EventType.GOAL_CREATED, EventType.GOAL_UPDATED):
            g = event.data(GoalPayload).goal
            if g.status == "active":
                self.goals[g.id] = g.model_dump(mode="json")
            else:
                self.goals.pop(g.id, None)
        elif t == EventType.ACTION_PROPOSED:
            self.proposals.append((self.now(), event.data(ActionPayload).spec))
        elif t == EventType.ACTION_APPROVED:
            p = event.data(ActionPayload)
            if p.spec.action == ActionType.WAIT:
                self._executed(p, {"outcome": "success", "detail": "stayed silent / kept observing"})
            elif p.spec.action == ActionType.NOTE:
                self._write_note(p)
        elif t in (EventType.ACTION_EXECUTED, EventType.ACTION_REJECTED):
            p = event.data(ActionPayload)
            self.pending.pop(p.decision_id, None)
            ok = t == EventType.ACTION_EXECUTED and p.result.get("outcome") == "success"
            if self.plan and self.plan["waiting"] == p.decision_id:
                if ok:
                    self.plan["waiting"] = None
                    self.plan["next_at"] = self.now() + self._hold(p.spec)
                else:
                    self.plan = None  # a step failed or was blocked: the rest of the plan is abandoned
            key = p.spec.action.value
            self.habits[key] = round(0.8 * self.habits.get(key, 0.6) + 0.2 * (1.0 if ok else 0.0), 4)
        elif t == EventType.WORKSPACE_UPDATED:
            if self.body.get("mode") in (Mode.AWAKE, Mode.DROWSY, "AWAKE", "DROWSY"):
                await self.decide(event.data(WorkspacePayload))

    # ------------------------------------------------------------------ decision
    async def decide(self, ws: WorkspacePayload) -> None:
        new_items = [i for i in ws.items if i.event_id in set(ws.new_item_ids)]
        modalities = sorted({i.modality.value for i in ws.items})
        cands: list[Candidate] = []
        addressed_pending = False

        for item in new_items:
            if item.event_type == EventType.UTTERANCE_UNDERSTOOD and item.event_id not in self.responded:
                self.responded.add(item.event_id)
                self.last_social = self.now()
                addressed_pending |= await self._utterance_candidates(item, cands)

        now = self.now()
        fresh = [(ts, s) for ts, s in self.proposals if now - ts < 30]
        self.proposals = []
        for _ts, spec in fresh:
            cands.append(Candidate(spec=spec, base=0.35 + 0.3 * spec.confidence, notes=["proposed by thought"]))

        if not any(c.spec.action in VOCAL_ACTIONS for c in cands):
            self._social_candidates(new_items, cands, now)
        if not any(c.spec.action in VOCAL_ACTIONS for c in cands):
            self._homeostatic_candidates(cands, now)

        if not cands:
            return
        replying_to = next((c.spec.about_event for c in cands if c.spec.action in VOCAL_ACTIONS), None)
        cands.append(Candidate(spec=ActionSpec(
            action=ActionType.WAIT, about_event=replying_to, confidence=0.9,
            reason="ignore what was said and keep observing" if replying_to else "nothing requires action",
            expected_outcome=EXPECTED[ActionType.WAIT]), base=0.2))
        for c in cands:
            c.spec.context_modalities = modalities
            c.utility = self.evaluate(c)
        await self._imagine(cands)

        chosen: list[Candidate] = []
        vocal = [c for c in cands if c.spec.action in VOCAL_ACTIONS or c.spec.action == ActionType.WAIT]
        other = [c for c in cands if c.spec.action not in VOCAL_ACTIONS and c.spec.action != ActionType.WAIT]
        best_vocal = max(vocal, key=lambda c: c.utility) if vocal else None
        if best_vocal and (best_vocal.spec.action != ActionType.WAIT or addressed_pending):
            chosen.append(best_vocal)
        if other:
            best_other = max(other, key=lambda c: c.utility)
            if best_other.utility > max(0.3, (best_vocal.utility - 0.5) if best_vocal else 0.3):
                chosen.append(best_other)
        vocal_choice = next((c for c in chosen if c.spec.action in VOCAL_ACTIONS), None)
        body_choice = next((c for c in chosen if c.spec.action not in VOCAL_ACTIONS and c.spec.action != ActionType.WAIT), None)
        if vocal_choice and body_choice:  # the voice knows what the body is doing ("like this?")
            vocal_choice.spec.params["doing"] = body_choice.spec.params.get("expression") or body_choice.spec.action.value
        for c in chosen:
            self._select(c, cands)

    async def _utterance_candidates(self, item: WorkspaceItem, cands: list[Candidate]) -> bool:
        u = UtterancePayload.model_validate(item.event["payload"])
        a = u.analysis
        if self.settings.thought.deliberate_before_speaking and self.bus.has_responder("thought.deliberate"):
            d = await self.ask_one("thought.deliberate", {"analysis": a.model_dump(), "event_id": item.event_id},
                                   default=None)
        else:
            d = None
        d = d or rule_deliberate(a, False).model_dump()
        urgency = float(item.event.get("urgency", 0.5))
        goal_id = next((gid for gid, g in self.goals.items() if g.get("related_event") == item.event_id), None)
        if d["respond"]:
            cands.append(Candidate(spec=ActionSpec(
                action=ActionType.SPEAK, about_event=item.event_id, goal_id=goal_id,
                params={"intention": d["intention"], "analysis": a.model_dump(), "thought": d["thought"],
                        "addressed_to": a.speaker},
                reason=d["thought"], confidence=d["confidence"], expected_outcome=EXPECTED[ActionType.SPEAK]),
                base=0.75 + 0.15 * urgency, notes=["respond to speech"]))
            if a.intent == "command" and a.command == "sequence" and a.steps:
                specs = [s for s in (self._command_spec(st["command"], st.get("arg"), a, item.event_id, in_plan=True)
                                     for st in a.steps) if s]
                self.plan = {"steps": deque(specs), "waiting": None, "next_at": self.now() + 0.5,
                             "about": item.event_id, "request": a.text}
            elif a.intent == "command" and a.command == "internet":
                spec = self._command_spec("internet", a.command_arg, a, item.event_id)
                perms = await self.ask_one("safety.permissions", {}, default={}) or {}
                if spec and perms.get("internet_read"):  # nothing to give up if it is already off
                    cands.append(Candidate(spec=spec, base=0.8, notes=["requested"]))
            elif a.intent == "command" and a.command in COMMAND_ACTIONS:
                spec = self._command_spec(a.command, a.command_arg, a, item.event_id)
                cands.append(Candidate(spec=spec, base=0.8, notes=["requested"]))
            return True
        cands.append(Candidate(spec=ActionSpec(action=ActionType.WAIT, about_event=item.event_id,
                                               params={"intention": "stay_silent"},
                                               reason=d["thought"], confidence=d["confidence"],
                                               expected_outcome="I keep listening without interrupting"),
                               base=0.6, notes=["deliberate silence"]))
        return True

    @staticmethod
    def _command_spec(command: str, arg: str | None, a, about: str, in_plan: bool = False) -> ActionSpec | None:
        if command == "internet":
            if arg != "off":
                return None  # it cannot grant itself access: the reply explains who can
            return ActionSpec(action=ActionType.REVOKE, about_event=about, params={"permission": "internet_read"},
                              reason=f"{a.speaker} asked me to go offline", confidence=0.9,
                              expected_outcome=EXPECTED[ActionType.REVOKE])
        if command == "count":
            return ActionSpec(action=ActionType.SPEAK, about_event=about,
                              params={"intention": "recite", "text": count_text(arg), "addressed_to": a.speaker},
                              reason=f"{a.speaker} asked me to count {arg or ''}".strip(), confidence=0.9,
                              expected_outcome=EXPECTED[ActionType.SPEAK])
        act = COMMAND_ACTIONS.get(command)
        if act is None:
            return None
        params: dict = {"from_user": True}
        if act == ActionType.REMEMBER:
            params["content"] = arg or a.text
        elif act == ActionType.SIMULATE:
            params["scenario"] = arg or a.text
        elif act == ActionType.NOTE:
            params["text"] = arg or a.text
        elif act == ActionType.SET_GOAL:
            params["description"] = arg or a.text
        elif act == ActionType.EXPRESS:
            params["expression"] = arg or "smile"
            if in_plan and params["expression"] != "neutral":
                params["duration"] = 60.0  # held until a later step changes it
        return ActionSpec(action=act, about_event=about, params=params, reason=f"{a.speaker} asked me to {command}",
                          confidence=0.8, expected_outcome=EXPECTED.get(act, ""))

    @staticmethod
    def _hold(spec: ActionSpec) -> float:
        """How long a finished step needs before the next one: let it be seen, or finish being heard."""
        if spec.action == ActionType.SPEAK:
            return 0.5 + 0.45 * len(str(spec.params.get("text", "")).split())
        return 1.2

    def _advance_plan(self, now: float) -> None:
        plan = self.plan
        if not plan or plan["waiting"] or now < plan["next_at"]:
            return
        if not plan["steps"]:
            self.plan = None
            return
        spec = plan["steps"].popleft()
        c = Candidate(spec=spec, base=0.9, utility=0.9, notes=[f"step of a plan: {truncate(plan['request'], 60)}"])
        plan["waiting"] = self._select(c, [])

    def _social_candidates(self, new_items: list[WorkspaceItem], cands: list[Candidate], now: float) -> None:
        """Someone came into view while nobody was talking to me: acknowledging them is an option."""
        for item in new_items:
            vision = item.event.get("payload", {}).get("vision") or {}
            if item.event_type != EventType.PERCEPTION or not vision.get("people"):
                continue
            if now - self.last_social < 120 or now - self.last_noticed < 300:
                return
            self.last_noticed = now
            cands.append(Candidate(spec=ActionSpec(
                action=ActionType.SPEAK, about_event=None, params={"intention": "notice_person"},
                reason="a person appeared in front of me and nobody is talking to me", confidence=0.6,
                expected_outcome="the person knows I have noticed them"), base=0.55, notes=["social orienting"]))
            return

    def _homeostatic_candidates(self, cands: list[Candidate], now: float) -> None:
        b = self.body
        fatigue = float(b.get("fatigue", 0.0))
        in_conversation = now - self.last_social < 60
        if fatigue > 0.7 and not in_conversation and not self._pending(ActionType.SLEEP):
            cands.append(Candidate(spec=ActionSpec(action=ActionType.SLEEP, reason=f"I am tired (fatigue {fatigue:.2f})",
                                                   confidence=0.8, goal_id="g_maintain",
                                                   expected_outcome=EXPECTED[ActionType.SLEEP]),
                                   base=0.4 + 0.5 * fatigue, notes=["homeostasis"]))

    def evaluate(self, c: Candidate) -> float:
        a = c.spec.action
        e = self.emotion
        u = c.base
        for g in self.goals.values():
            if a.value in g.get("required_actions", []) or c.spec.goal_id == g.get("id"):
                u += 0.15 * float(g.get("priority", 0.5))
        fear, frus = e.get("fear", 0.0), e.get("frustration", 0.0)
        if a in (ActionType.SPEAK, ActionType.LOOK, ActionType.NOTE, ActionType.MOVE):
            u -= 0.15 * fear
        if a in (ActionType.ASK, ActionType.WAIT):
            u += 0.05 * fear
        if a == ActionType.SPEAK:
            u += 0.08 * e.get("social", 0.2)
        if a in (ActionType.INVESTIGATE, ActionType.SIMULATE, ActionType.LOOK):
            u += 0.15 * e.get("boredom", 0.0) + 0.1 * e.get("curiosity", 0.35)
        habit = self.habits.get(a.value, 0.6)
        u += 0.1 * (habit - 0.6) - 0.1 * frus * (1.0 - habit)
        energy = float(self.body.get("energy", 1.0))
        u -= ENERGY_COST.get(a, 0.05) * (1.2 - energy)
        return round(u, 4)

    async def _imagine(self, cands: list[Candidate]) -> None:
        if not self.bus.has_responder("imagination.simulate"):
            return
        ranked = sorted(cands, key=lambda c: -c.utility)
        top = ranked[: self.settings.imagination.max_candidates]
        if len(top) < 2 or (top[0].utility - top[1].utility > 0.4 and top[0].spec.action not in VOCAL_ACTIONS):
            return
        for c in top:
            sim = await self.ask_one("imagination.simulate", {"spec": c.spec.model_dump(mode="json")}, default=None)
            if not sim:
                continue
            c.spec.simulated = True
            c.spec.expected_outcome = sim.get("predicted_outcome", c.spec.expected_outcome)
            c.utility = round(c.utility + 0.3 * float(sim.get("value", 0.0)) - 0.3 * len(sim.get("risks", [])) * 0.3, 4)
            c.notes.append(f"imagined value {float(sim.get('value', 0)):.2f}")

    def _pending(self, action: ActionType) -> bool:
        return any(s.action == action for s, _ts in self.pending.values())

    def _select(self, c: Candidate, all_cands: list[Candidate]) -> str:
        did = new_id("d_")
        c.spec.value = c.utility
        self.pending[did] = (c.spec, self.now())
        alternatives = sorted(({"action": o.spec.action.value, "utility": o.utility} for o in all_cands if o is not c),
                              key=lambda x: -x["utility"])[:4]
        self.decisions = ([{"t": self.now(), "action": c.spec.action.value, "reason": truncate(c.spec.reason, 140),
                            "utility": c.utility, "confidence": c.spec.confidence, "simulated": c.spec.simulated,
                            "expected": c.spec.expected_outcome, "alternatives": alternatives, "notes": c.notes}]
                          + self.decisions)[:40]
        self.emit(EventType.ACTION_SELECTED,
                  ActionPayload(decision_id=did, spec=c.spec, status="selected", reason=c.spec.reason,
                                result={"alternatives": alternatives}),
                  summary=f"selected {c.spec.action.value} (u={c.utility:.2f}): {truncate(c.spec.reason, 100)}",
                  confidence=c.spec.confidence, modality=Modality.MOTOR)
        return did

    # ------------------------------------------------------------------ simple effectors
    def _executed(self, p: ActionPayload, result: dict) -> None:
        self.emit(EventType.ACTION_EXECUTED,
                  ActionPayload(decision_id=p.decision_id, spec=p.spec, status="executed", result=result),
                  summary=f"Executed {p.spec.action.value}: {result.get('detail', result.get('outcome'))}",
                  modality=Modality.MOTOR)

    def _write_note(self, p: ActionPayload) -> None:
        text = str(p.spec.params.get("text") or p.spec.reason)[:2000]
        try:
            self.notebook.parent.mkdir(parents=True, exist_ok=True)
            with self.notebook.open("a", encoding="utf-8") as f:
                f.write(f"- [{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.now()))}] {text}\n")
            self._executed(p, {"outcome": "success", "detail": f"wrote note to {self.notebook.name}"})
        except OSError as exc:
            self._executed(p, {"outcome": "failed", "detail": str(exc)})

    def snapshot(self) -> dict:
        plan = [s.action.value + (f":{s.params.get('expression')}" if s.params.get("expression") else "")
                for s in self.plan["steps"]] if self.plan else None
        return {"decisions": self.decisions[:12], "pending": [s.action.value for s, _ in self.pending.values()],
                "plan": plan,
                "habits": self.habits}

    def trace_state(self) -> dict:
        return {"pending": len(self.pending), "habits": self.habits}


async def _true(_q: dict) -> bool:
    return True
