"""Base class for cognitive modules and the shared (read-only) organism context."""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel

from .events import Event, EventType, make_event

if TYPE_CHECKING:
    from ..config import Settings
    from ..models.embeddings import Embedder
    from ..models.llm import LLMRouter
    from ..persistence.store import Store
    from .bus import Bus
    from .clock import Clock, TemporalState


@dataclass
class Context:
    settings: "Settings"
    bus: "Bus"
    clock: "Clock"
    store: "Store"
    llm: "LLMRouter"
    embedder: "Embedder"          # semantic embeddings (may be a neural model)
    fast_embedder: "Embedder"     # cheap hashing embeddings for attention/novelty
    temporal: "TemporalState"
    rng: random.Random
    enabled_modules: set[str] = field(default_factory=set)

    @property
    def ltm_enabled(self) -> bool:
        return "memory" in self.enabled_modules


class CognitiveModule:
    name: ClassVar[str] = "module"
    subscriptions: ClassVar[tuple[str, ...]] = ()

    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.bus = ctx.bus
        self.settings = ctx.settings
        self.log = logging.getLogger(f"organism.{self.name}")

    # lifecycle ---------------------------------------------------------------
    async def start(self) -> None:
        """Register responders, restore persisted state, start background loops."""

    async def stop(self) -> None:
        """Persist state and stop background loops."""

    async def handle(self, event: Event) -> None:
        """React to a subscribed event."""

    # observability -----------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        """Full state for the dashboard."""
        return {}

    def trace_state(self) -> dict[str, Any]:
        """Compact state used for before/after transition records."""
        return {}

    # helpers -----------------------------------------------------------------
    def now(self) -> float:
        return self.ctx.clock.now()

    def emit(self, etype: EventType, payload: BaseModel | dict | None = None, **meta: Any) -> Event:
        return self.bus.publish(make_event(etype, self.name, payload, **meta))

    def respond(self, topic: str, fn) -> None:
        self.bus.register_responder(topic, self.name, fn)

    async def ask(self, topic: str, query: dict | None = None, timeout: float = 30.0):
        return await self.bus.request(topic, query, timeout)

    async def ask_one(self, topic: str, query: dict | None = None, default: Any = None,
                      timeout: float = 30.0) -> Any:
        return await self.bus.request_one(topic, query, default, timeout)

    def kv_load(self, key: str, default: Any = None) -> Any:
        return self.ctx.store.kv_get(f"{self.name}.{key}", default)

    def kv_save(self, key: str, value: Any) -> None:
        self.ctx.store.kv_set(f"{self.name}.{key}", value)
