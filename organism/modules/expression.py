"""Facial expression effector (the face's motor system).

``express`` actions (asked for by a person, or chosen by the organism) set a timed facial
expression that the face renders on top of its emotion-driven baseline. Smiling feeds back into
the valuation system a little (a facial-feedback-like loop): the face is part of the body, not
just a display.
"""
from __future__ import annotations

from ..core.events import ActionPayload, ActionType, Event, EventType, Modality
from ..core.module import CognitiveModule
from ..core.util import new_id

EXPRESSIONS = {
    "smile": 6.0, "big_smile": 6.0, "laugh": 3.5, "sad": 6.0, "angry": 5.0, "surprised": 3.5, "wink": 1.2,
    "eyes_closed": 12.0, "raise_eyebrows": 3.0, "look_left": 3.0, "look_right": 3.0, "look_up": 3.0,
    "look_down": 3.0, "neutral": 0.5, "thinking": 4.0,
}
DESCRIPTION = {
    "smile": "smiling", "big_smile": "smiling broadly", "laugh": "laughing", "sad": "making a sad face",
    "angry": "making an angry face", "surprised": "looking surprised", "wink": "winking",
    "eyes_closed": "keeping my eyes closed", "raise_eyebrows": "raising my eyebrows", "look_left": "looking left",
    "look_right": "looking right", "look_up": "looking up", "look_down": "looking down",
    "neutral": "relaxing my face", "thinking": "making a thinking face",
}


class Expression(CognitiveModule):
    name = "expression"
    subscriptions = (EventType.ACTION_APPROVED,)

    def __init__(self, ctx):
        super().__init__(ctx)
        self.current: dict | None = None

    async def start(self) -> None:
        self.respond("executor.express", _true)
        self.respond("expression.current", self._current)

    async def _current(self, _q: dict) -> dict | None:
        return self.view()

    def view(self) -> dict | None:
        c = self.current
        if not c:
            return None
        remaining = c["until"] - self.now()
        if remaining <= 0:
            return None
        return {"name": c["name"], "id": c["id"], "remaining": round(remaining, 2),
                "description": DESCRIPTION.get(c["name"], c["name"])}

    async def handle(self, event: Event) -> None:
        p = event.data(ActionPayload)
        if p.spec.action != ActionType.EXPRESS:
            return
        name = str(p.spec.params.get("expression", "smile"))
        if name not in EXPRESSIONS:
            name = "smile"
        duration = float(p.spec.params.get("duration", EXPRESSIONS[name]))
        self.current = {"name": name, "id": new_id("x_"), "until": self.now() + duration}
        self.emit(EventType.ACTION_EXECUTED,
                  ActionPayload(decision_id=p.decision_id, spec=p.spec, status="executed",
                                result={"outcome": "success", "expression": name, "duration": duration,
                                        "detail": DESCRIPTION.get(name, name)}),
                  summary=f"Executed express: {DESCRIPTION.get(name, name)}", modality=Modality.MOTOR)

    def snapshot(self) -> dict:
        return {"current": self.view()}

    def trace_state(self) -> dict:
        v = self.view()
        return {"expression": v["name"] if v else None}


async def _true(_q: dict) -> bool:
    return True
