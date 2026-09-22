"""Organism time and temporal continuity."""
from __future__ import annotations

import time

from pydantic import BaseModel

from .util import new_id


class Clock:
    """Real mode follows (optionally scaled) wall time; virtual mode only moves when advanced.

    Organism time is always an epoch timestamp so memories remain meaningful across restarts.
    """

    def __init__(self, mode: str = "real", time_scale: float = 1.0, start: float | None = None):
        self.mode = mode
        self.scale = time_scale
        self._real0 = time.time()
        self._org0 = start if start is not None else self._real0
        self._virtual = self._org0

    @property
    def virtual(self) -> bool:
        return self.mode == "virtual"

    def now(self) -> float:
        if self.virtual:
            return self._virtual
        return self._org0 + (time.time() - self._real0) * self.scale

    def advance(self, seconds: float) -> None:
        if self.virtual:
            self._virtual += seconds
        else:
            self._org0 += seconds


class TemporalState(BaseModel):
    """The organism's sense of time: where it is in its own history."""

    boot_time: float = 0.0
    current_time: float = 0.0
    previous_time: float = 0.0
    elapsed_since_boot: float = 0.0
    last_shutdown: float | None = None
    downtime: float | None = None
    episode_id: str = ""
    episode_started: float = 0.0
    sleep_cycles: int = 0
    last_external_input: float | None = None

    def new_episode(self, now: float) -> str:
        self.episode_id = new_id("ep_")
        self.episode_started = now
        return self.episode_id
