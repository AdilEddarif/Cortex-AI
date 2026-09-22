"""Global workspace: the small set of contents that is globally broadcast each cycle.

Items that win the attention competition enter the workspace; existing items decay and
are displaced when capacity is exceeded. Every cycle a WORKSPACE_UPDATED broadcast reaches
all consumers (memory, thought, language, action, emotion, self-model ...). Consumers read
the broadcast; they cannot write to the workspace directly (only nominate).

``LocalRelay`` replaces the workspace in the "no global workspace" ablation: every
nominated event is relayed on its own, immediately, with no competition, no capacity limit
and no co-present context (``integrated=False``).
"""
from __future__ import annotations

from collections import deque

from ..core.events import (
    AttentionPayload, Event, EventType, Modality, TickPayload, WorkspaceItem, WorkspacePayload,
)
from ..core.module import CognitiveModule


def _item_from_event(ev: dict, score: float, cycle: int, components: dict | None = None) -> WorkspaceItem:
    return WorkspaceItem(
        event_id=ev["id"], event_type=ev["type"], source=ev["source"],
        modality=ev.get("modality", Modality.INTERNAL), summary=ev.get("summary", ""), score=score,
        entered_cycle=cycle, components=components or {}, event=ev,
    )


class Workspace(CognitiveModule):
    name = "workspace"

    def __init__(self, ctx):
        super().__init__(ctx)
        self.cfg = self.settings.workspace
        self.items: list[WorkspaceItem] = []
        self.history: deque[dict] = deque(maxlen=200)
        self.cycle = 0
        # Without an attention module the workspace falls back to pure recency.
        self.recency_mode = not ctx.settings.enabled("attention")
        self._recency_pool: list[Event] = []

    @property
    def subscriptions(self):  # type: ignore[override]
        return ("*",) if self.recency_mode else (EventType.ATTENTION_CHANGED,)

    async def start(self) -> None:
        self.respond("workspace.current", self._current)

    async def _current(self, _q: dict) -> dict:
        return {"cycle": self.cycle, "items": [
            {"event_id": i.event_id, "type": i.event_type.value, "source": i.source, "modality": i.modality.value,
             "summary": i.summary, "score": round(i.score, 3)} for i in self.items]}

    async def handle(self, event: Event) -> None:
        if event.type == EventType.ATTENTION_CHANGED:
            p = event.data(AttentionPayload)
            winners = [(_item_from_event(s.event, s.score, self.bus.cycle, s.components)) for s in p.selected]
            self.update(self.bus.cycle, winners)
        elif self.recency_mode:
            if event.type == EventType.TICK:
                cycle = event.data(TickPayload).cycle
                latest = self._recency_pool[-self.cfg.capacity:]
                self._recency_pool.clear()
                self.update(cycle, [_item_from_event(e.model_dump(mode="json"), 1.0, cycle) for e in latest])
            elif event.nominated:
                self._recency_pool.append(event)

    def update(self, cycle: int, winners: list[WorkspaceItem]) -> None:
        self.cycle = cycle
        for it in self.items:
            it.score *= self.cfg.decay
        existing = {i.event_id for i in self.items}
        new = [w for w in winners if w.event_id not in existing]
        merged = sorted(self.items + new, key=lambda i: -i.score)
        merged = [i for i in merged if i.score >= self.cfg.floor][: self.cfg.capacity]
        kept_ids = {i.event_id for i in merged}
        new_ids = [w.event_id for w in new if w.event_id in kept_ids]
        self.items = merged
        focus = merged[0].summary if merged else None
        if new_ids:
            self.history.append({"cycle": cycle, "t": self.now(), "focus": focus,
                                 "new": [i.summary for i in merged if i.event_id in new_ids]})
        self.emit(EventType.WORKSPACE_UPDATED,
                  WorkspacePayload(cycle=cycle, items=merged, new_item_ids=new_ids, focus=focus),
                  summary=f"workspace[{len(merged)}] focus: {(focus or '-')[:100]}")

    def snapshot(self) -> dict:
        return {
            "cycle": self.cycle,
            "items": [{"summary": i.summary, "type": i.event_type.value, "source": i.source,
                       "modality": i.modality.value, "score": round(i.score, 3),
                       "age": self.cycle - i.entered_cycle} for i in self.items],
            "stream": list(self.history)[-20:],
            "mode": "recency" if self.recency_mode else "attention",
        }

    def trace_state(self) -> dict:
        return {"items": [i.event_id for i in self.items]}


class LocalRelay(CognitiveModule):
    """Ablation stand-in for the global workspace (Experiment C)."""

    name = "workspace_bypass"
    subscriptions = ("*",)

    async def handle(self, event: Event) -> None:
        if not event.nominated:
            return
        item = _item_from_event(event.model_dump(mode="json"), event.salience, self.bus.cycle)
        self.emit(EventType.WORKSPACE_UPDATED,
                  WorkspacePayload(cycle=self.bus.cycle, items=[item], new_item_ids=[item.event_id],
                                   focus=item.summary, integrated=False),
                  summary=f"local relay: {event.summary[:100]}")

    def snapshot(self) -> dict:
        return {"mode": "ablated (local relay, no global broadcast)"}
