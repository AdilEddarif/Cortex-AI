"""Policy / safety layer between cognition and execution.

    proposed action -> ACTION_SELECTED -> [policy checks] -> ACTION_APPROVED | ACTION_REJECTED

Checks: action type allow-list, effector availability (capability), explicit permissions for
actions that touch the outside world, rate limits, no external action while asleep, parameter
sanity. ``safety.check_utterance`` filters outgoing speech (secret redaction, length).
Rejections are nominated so the organism notices (and learns about) its own limits.
"""
from __future__ import annotations

from collections import deque

from ..core.events import (
    EXTERNAL_ACTIONS, ActionPayload, ActionType, CommandPayload, Event, EventType, Modality, Mode, SelfStatePayload,
    TickPayload,
)
from ..core.module import CognitiveModule
from ..core.redact import contains_secret, redact_text


class Safety(CognitiveModule):
    name = "safety"
    subscriptions = (EventType.ACTION_SELECTED, EventType.TICK, EventType.COMMAND)

    def __init__(self, ctx):
        super().__init__(ctx)
        self.cfg = self.settings.safety
        self.mode = Mode.AWAKE
        self._speech_times: deque[float] = deque(maxlen=100)
        self.log_entries: deque[dict] = deque(maxlen=40)
        self.counts = {"approved": 0, "rejected": 0}

    async def start(self) -> None:
        self.respond("safety.check_utterance", self._check_utterance)
        self.respond("safety.permissions", self._permissions)
        self.respond("executor.revoke", _true)

    async def _permissions(self, _q: dict) -> dict:
        return dict(self.cfg.permissions)

    def _set_permission(self, name: str, value: bool, by: str) -> None:
        before = self.cfg.permissions.get(name, False)
        self.cfg.permissions[name] = value
        self.log_entries.appendleft({"t": self.now(), "action": f"{'grant' if value else 'revoke'} {name}",
                                     "approved": True, "reason": f"by {by}"})
        if before != value:
            self.emit(EventType.SELF_STATE_CHANGED,
                      SelfStatePayload(changes={"permission": name, "value": value},
                                       reason=f"{by} {'granted' if value else 'revoked'} {name}"),
                      summary=f"{by} {'granted me' if value else 'revoked my'} {name.replace('_', ' ')} permission",
                      nominated=True, salience=0.4)

    async def _check_utterance(self, q: dict) -> dict:
        text = str(q.get("text", ""))
        issues = []
        if contains_secret(text):
            issues.append("secret-like content redacted")
            text = redact_text(text)
        if len(text) > self.cfg.max_utterance_chars:
            issues.append("truncated")
            text = text[: self.cfg.max_utterance_chars]
        return {"text": text, "allowed": True, "issues": issues}

    async def handle(self, event: Event) -> None:
        if event.type == EventType.TICK:
            self.mode = event.data(TickPayload).body.mode
            return
        if event.type == EventType.COMMAND:  # the operator's channel, not something said to the organism
            c = event.data(CommandPayload)
            name = str(c.args.get("permission", ""))
            if c.command in ("grant", "revoke") and name:
                self._set_permission(name, c.command == "grant", "the operator")
            return
        p = event.data(ActionPayload)
        reason = self.check(p)
        entry = {"t": self.now(), "action": p.spec.action.value, "approved": reason is None, "reason": reason}
        self.log_entries.appendleft(entry)
        if reason is None:
            self.counts["approved"] += 1
            if p.spec.action in (ActionType.SPEAK, ActionType.ASK):
                self._speech_times.append(self.now())
            self.emit(EventType.ACTION_APPROVED,
                      ActionPayload(decision_id=p.decision_id, spec=p.spec, status="approved"),
                      summary=f"approved {p.spec.action.value}", confidence=p.spec.confidence)
            if p.spec.action == ActionType.REVOKE:  # giving up a permission is always allowed
                name = str(p.spec.params.get("permission", ""))
                self._set_permission(name, False, "I")
                self.emit(EventType.ACTION_EXECUTED,
                          ActionPayload(decision_id=p.decision_id, spec=p.spec, status="executed",
                                        result={"outcome": "success", "detail": f"gave up {name}"}),
                          summary=f"Executed revoke: gave up {name}", modality=Modality.MOTOR)
        else:
            self.counts["rejected"] += 1
            self.emit(EventType.ACTION_REJECTED,
                      ActionPayload(decision_id=p.decision_id, spec=p.spec, status="rejected", reason=reason),
                      summary=f"I could not {p.spec.action.value}: {reason}", nominated=True, salience=0.45,
                      confidence=0.95)

    def check(self, p: ActionPayload) -> str | None:
        a = p.spec.action
        if a.value not in self.cfg.allowed_actions:
            return f"'{a.value}' is not allowed by my action policy"
        if a != ActionType.WAIT and not self.bus.has_responder(f"executor.{a.value}"):
            if a == ActionType.MOVE:
                return "I have no body or motor effectors, so I cannot move"
            if a == ActionType.LOOK:
                return "I have no working visual sensor to look with"
            return f"no effector is available for '{a.value}'"
        perm = self.cfg.action_permissions.get(a.value)
        if perm and not self.cfg.permissions.get(perm, False):
            return f"'{a.value}' requires the '{perm}' permission, which has not been granted"
        if a in EXTERNAL_ACTIONS and self.mode in (Mode.ASLEEP, Mode.DREAMING):
            return "external actions are disabled while I am asleep"
        if a in (ActionType.SPEAK, ActionType.ASK):
            now = self.now()
            recent = [t for t in self._speech_times if now - t < 60]
            if len(recent) >= self.speech_limit():
                return "speech rate limit reached"
        if a == ActionType.NOTE and len(str(p.spec.params.get("text", ""))) > 2000:
            return "note too long"
        return None

    def speech_limit(self) -> int:
        return self.settings.speech.max_per_minute

    def snapshot(self) -> dict:
        return {"counts": self.counts, "recent": list(self.log_entries)[:12],
                "permissions": self.cfg.permissions,
                "policy": {"allowed": self.cfg.allowed_actions, "permissions_required": self.cfg.action_permissions}}

    def trace_state(self) -> dict:
        return dict(self.counts)


async def _true(_q: dict) -> bool:
    return True
