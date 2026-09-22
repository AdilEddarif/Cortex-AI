"""Reproducibility log: every cognitive transition (module x input event) that produced
output or changed module state is recorded with before/after state, models and latency."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from .events import Event

_MAX_STATE_CHARS = 6000


def _dump(state: Any) -> str:
    try:
        text = json.dumps(state, default=str, sort_keys=True)
    except Exception:
        text = repr(state)
    return text[:_MAX_STATE_CHARS]


class TransitionRecorder:
    def __init__(self, store=None, enabled: bool = True, capture_state: bool = True):
        self.store = store
        self.enabled = enabled and store is not None
        self.capture_state = capture_state
        self.count = 0

    def record(self, *, module: str, event: Event, outputs: list[str], before: Any, after: Any,
               models: list[str], latency_ms: float) -> None:
        b = _dump(before) if before is not None else None
        a = _dump(after) if after is not None else None
        changed = b != a
        if not outputs and not changed:
            return  # nothing happened: keep the log lean
        self.count += 1
        self.store.log_transition({
            "event_id": event.id,
            "ts": event.timestamp,
            "module": module,
            "input_type": event.type.value,
            "input_summary": event.summary[:300],
            "outputs": json.dumps(outputs),
            "state_before": b if changed else None,
            "state_after": a if changed else None,
            "state_digest": hashlib.sha1((a or "").encode()).hexdigest()[:12],
            "model_used": ",".join(sorted(set(models))) or None,
            "latency_ms": round(latency_ms, 3),
            "confidence": event.confidence,
        })
