"""Thalamus-inspired routing layer.

Modules never call each other directly. They:
  * ``publish`` typed events (routed by type to subscribers' private inboxes),
  * ``nominate`` items for attention (``event.nominated = True``),
  * ``request`` information from registered responders (``memory.recall`` ...).

The bus stamps provenance (``caused_by``), time and cycle on every event, applies sensory
gating (arousal-dependent thalamic gain), persists the event log, and tracks in-flight work
so that tests and experiments can deterministically wait for quiescence (``settle``).
"""
from __future__ import annotations

import asyncio
import contextvars
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from .events import Event, EventType, Modality, make_event, ErrorPayload

log = logging.getLogger("organism.bus")

Handler = Callable[[Event], Awaitable[None]]
Responder = Callable[[dict], Awaitable[Any]]

# Context propagated through handler execution (asyncio copies it into spawned tasks).
current_cause: contextvars.ContextVar[str | None] = contextvars.ContextVar("current_cause", default=None)
current_outputs: contextvars.ContextVar[list | None] = contextvars.ContextVar("current_outputs", default=None)
current_model: contextvars.ContextVar[list | None] = contextvars.ContextVar("current_model", default=None)

WILDCARD = "*"


@dataclass
class Response:
    source: str
    data: Any


@dataclass
class Inbox:
    name: str
    handler: Handler
    queue: asyncio.Queue
    task: asyncio.Task | None = None
    processed: int = 0
    dropped: int = 0
    errors: int = 0
    busy_s: float = 0.0
    snapshot: Callable[[], dict] | None = None


@dataclass
class BusStats:
    published: int = 0
    by_type: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    requests: int = 0


class Bus:
    def __init__(self, clock, store=None, recorder=None, max_queue: int = 2000):
        self.clock = clock
        self.store = store
        self.recorder = recorder
        self.max_queue = max_queue
        self.cycle = 0
        self.sensory_gain: dict[str, float] = {}
        self._subs: dict[str, list[Inbox]] = defaultdict(list)
        self._inboxes: dict[str, Inbox] = {}
        self._responders: dict[str, list[tuple[str, Responder]]] = defaultdict(list)
        self._observers: list[Callable[[Event], None]] = []
        self._inflight = 0
        self._idle = asyncio.Event()
        self._idle.set()
        self._tasks: set[asyncio.Task] = set()
        self._running = False
        self.stats = BusStats()

    # ------------------------------------------------------------------ registration
    def attach(self, name: str, handler: Handler, types: list[str], snapshot=None) -> Inbox:
        inbox = Inbox(name=name, handler=handler, queue=asyncio.Queue(), snapshot=snapshot)
        self._inboxes[name] = inbox
        for t in types:
            self._subs[str(t)].append(inbox)
        if self._running:
            inbox.task = asyncio.create_task(self._worker(inbox), name=f"inbox:{name}")
        return inbox

    def register_responder(self, topic: str, source: str, fn: Responder) -> None:
        self._responders[topic].append((source, fn))

    def has_responder(self, topic: str) -> bool:
        return bool(self._responders.get(topic))

    def topics(self) -> list[str]:
        return sorted(t for t, v in self._responders.items() if v)

    def add_observer(self, fn: Callable[[Event], None]) -> None:
        self._observers.append(fn)

    def remove_observer(self, fn: Callable[[Event], None]) -> None:
        if fn in self._observers:
            self._observers.remove(fn)

    async def start(self) -> None:
        self._running = True
        for inbox in self._inboxes.values():
            if inbox.task is None:
                inbox.task = asyncio.create_task(self._worker(inbox), name=f"inbox:{inbox.name}")

    async def stop(self) -> None:
        self._running = False
        tasks = [i.task for i in self._inboxes.values() if i.task] + list(self._tasks)
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    # ------------------------------------------------------------------ publishing
    def publish(self, event: Event) -> Event:
        if not event.timestamp:
            event.timestamp = self.clock.now()
        event.cycle = self.cycle
        cause = current_cause.get()
        if cause and cause != event.id and cause not in event.caused_by:
            event.caused_by.append(cause)
        outs = current_outputs.get()
        if outs is not None:
            outs.append(event.id)

        if event.type == EventType.PERCEPTION and not event.payload.get("self_generated"):
            gain = self.sensory_gain.get(event.modality.value, self.sensory_gain.get("*", 1.0))
            event.gain = gain
            event.salience = round(event.salience * gain, 4)

        self.stats.published += 1
        self.stats.by_type[event.type.value] += 1
        if self.store is not None:
            self.store.log_event(event)
        for obs in list(self._observers):
            try:
                obs(event)
            except Exception:  # observers must never break cognition
                log.exception("observer failed")

        seen: set[str] = set()
        for inbox in self._subs.get(event.type.value, []) + self._subs.get(WILDCARD, []):
            if inbox.name == event.source or inbox.name in seen:
                continue
            seen.add(inbox.name)
            self._enqueue(inbox, event)
        return event

    def emit(self, etype: EventType, source: str, payload=None, **meta) -> Event:
        return self.publish(make_event(etype, source, payload, **meta))

    def _enqueue(self, inbox: Inbox, event: Event) -> None:
        if inbox.queue.qsize() >= self.max_queue:
            try:
                inbox.queue.get_nowait()
                inbox.dropped += 1
                self._done()
            except asyncio.QueueEmpty:
                pass
        self._inflight += 1
        self._idle.clear()
        inbox.queue.put_nowait(event)

    def _done(self) -> None:
        self._inflight -= 1
        if self._inflight <= 0:
            self._inflight = 0
            self._idle.set()

    async def _worker(self, inbox: Inbox) -> None:
        while True:
            event = await inbox.queue.get()
            try:
                await self._dispatch(inbox, event)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                inbox.errors += 1
                log.exception("module %s failed on %s", inbox.name, event.type)
                if event.type != EventType.MODULE_ERROR:
                    self.emit(
                        EventType.MODULE_ERROR, inbox.name,
                        ErrorPayload(module=inbox.name, error=repr(exc)[:500], event_id=event.id),
                        summary=f"Module {inbox.name} failed: {exc!r}"[:200],
                        nominated=inbox.errors <= 3, salience=0.4,
                    )
            finally:
                self._done()

    async def _dispatch(self, inbox: Inbox, event: Event) -> None:
        tok_c = current_cause.set(event.id)
        outputs: list[str] = []
        tok_o = current_outputs.set(outputs)
        models: list[str] = []
        tok_m = current_model.set(models)
        record = self.recorder is not None and self.recorder.enabled
        before = inbox.snapshot() if (record and inbox.snapshot and self.recorder.capture_state) else None
        t0 = time.perf_counter()
        try:
            await inbox.handler(event)
        finally:
            dt = time.perf_counter() - t0
            inbox.processed += 1
            inbox.busy_s += dt
            current_cause.reset(tok_c)
            current_outputs.reset(tok_o)
            current_model.reset(tok_m)
            if record:
                after = inbox.snapshot() if (inbox.snapshot and self.recorder.capture_state) else None
                self.recorder.record(
                    module=inbox.name, event=event, outputs=outputs, before=before, after=after,
                    models=models, latency_ms=dt * 1000.0,
                )

    # ------------------------------------------------------------------ request/response
    async def request(self, topic: str, query: dict | None = None, timeout: float = 30.0) -> list[Response]:
        handlers = list(self._responders.get(topic, []))
        if not handlers:
            return []
        self.stats.requests += 1

        async def call(source: str, fn: Responder):
            try:
                return Response(source, await asyncio.wait_for(fn(dict(query or {})), timeout))
            except asyncio.TimeoutError:
                log.warning("request %s to %s timed out", topic, source)
            except Exception:
                log.exception("responder %s for %s failed", source, topic)
            return None

        results = await asyncio.gather(*(call(s, f) for s, f in handlers))
        return [r for r in results if r is not None and r.data is not None]

    async def request_one(self, topic: str, query: dict | None = None, default: Any = None,
                          timeout: float = 30.0) -> Any:
        res = await self.request(topic, query, timeout)
        return res[0].data if res else default

    # ------------------------------------------------------------------ background work
    def spawn(self, coro, name: str | None = None) -> asyncio.Task:
        """Run background cognition that counts towards in-flight work (see ``settle``)."""
        self._inflight += 1
        self._idle.clear()
        task = asyncio.create_task(coro, name=name)
        self._tasks.add(task)

        def _finished(t: asyncio.Task) -> None:
            self._tasks.discard(t)
            self._done()
            if not t.cancelled() and t.exception() is not None:
                log.error("background task %s failed: %r", name, t.exception())

        task.add_done_callback(_finished)
        return task

    async def settle(self, timeout: float = 60.0) -> None:
        """Wait until no events are queued and no handler or background task is running."""
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"bus did not settle (in-flight={self._inflight})")
            await asyncio.wait_for(self._idle.wait(), remaining)
            await asyncio.sleep(0)
            if self._inflight == 0:
                return

    # ------------------------------------------------------------------ observability
    def topology(self) -> dict:
        """Who listens to what, and who answers which requests (the organism's wiring diagram)."""
        subs: dict[str, list[str]] = {}
        for etype, inboxes in self._subs.items():
            subs[etype] = sorted({i.name for i in inboxes})
        return {"modules": sorted(self._inboxes), "subscriptions": subs,
                "responders": {t: sorted({s for s, _ in v}) for t, v in self._responders.items() if v}}

    def module_stats(self) -> dict[str, dict]:
        return {
            n: {"processed": i.processed, "queued": i.queue.qsize(), "dropped": i.dropped,
                "errors": i.errors, "busy_ms": round(i.busy_s * 1000, 1)}
            for n, i in self._inboxes.items()
        }

    def gain_for(self, modality: Modality) -> float:
        return self.sensory_gain.get(modality.value, self.sensory_gain.get("*", 1.0))
