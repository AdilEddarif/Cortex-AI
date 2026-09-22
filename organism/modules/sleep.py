"""Offline processing during sleep.

ASLEEP (NREM-like): replay recent, important, unconsolidated episodic memories; cluster
similar episodes; abstract them into semantic knowledge; strengthen replayed traces; prune
weak ones; let the self-model rewrite its narrative and the world model decay stale beliefs.

DREAMING (REM-like): sample weighted memory fragments and recombine them (imagination) into
dream-like simulations that are fed to the workspace while external input is gated. They are
stored as ``dream`` memories, never as real episodes.
"""
from __future__ import annotations

from collections import Counter

from pydantic import BaseModel, Field

from ..core.events import (
    ConsolidationPayload, DreamPayload, Event, EventType, MemoryRef, MemoryReplayedPayload, Modality, Mode,
    TickPayload,
)
from ..core.module import CognitiveModule
from ..core.util import content_words, new_id, truncate

_GENERIC = {"said", "user", "thought", "heard", "saw", "perceived", "surprised", "adopted", "goal", "someone",
            "noted", "remember", "achieved", "respond"}


class Abstraction(BaseModel):
    facts: list[str] = Field(default_factory=list)


def rule_abstract(contents: list[str]) -> Abstraction:
    facts: list[str] = []
    speakers = Counter()
    topics: Counter = Counter()   # number of memories each word occurs in
    seen: Counter = Counter()
    for c in contents:
        if " said to me:" in c:
            speakers[c.split(" said to me:")[0].strip()] += 1
            topics.update({w for w in content_words(c.split(":", 1)[1]) if w not in _GENERIC})
        elif c.startswith("I saw:"):
            seen.update({w for w in content_words(c[6:]) if w not in _GENERIC})
        elif not c.startswith(("I said", "I thought", "I was surprised")):
            topics.update({w for w in content_words(c) if w not in _GENERIC})
    topics = Counter({w: n for w, n in topics.items() if n >= 2})
    seen = Counter({w: n for w, n in seen.items() if n >= 2})
    for who, n in speakers.items():
        if n >= 2:
            about = ", ".join(w for w, _ in topics.most_common(3))
            facts.append(f"I have talked with {who} several times" + (f", about {about}." if about else "."))
    if seen:
        facts.append(f"I have repeatedly seen: {', '.join(w for w, _ in seen.most_common(4))}.")
    if not facts and topics:
        facts.append(f"A recurring theme in my experience: {', '.join(w for w, _ in topics.most_common(3))}.")
    return Abstraction(facts=facts)


class Sleep(CognitiveModule):
    name = "sleep"
    subscriptions = (EventType.TICK, EventType.SLEEP_STARTED, EventType.SLEEP_ENDED, EventType.DREAM_STARTED)

    def __init__(self, ctx):
        super().__init__(ctx)
        self.cfg = self.settings.sleep
        self.stats = ConsolidationPayload()
        self._started: float | None = None
        self._busy = False
        self._maintenance_done = False
        self.dream_id: str | None = None
        self.dream_step = 0
        self.dreams: list[dict] = []
        self.history: list[dict] = []

    async def handle(self, event: Event) -> None:
        t = event.type
        if t == EventType.SLEEP_STARTED:
            self.stats = ConsolidationPayload()
            self._started = self.now()
            self._maintenance_done = False
        elif t == EventType.DREAM_STARTED:
            self.dream_id, self.dream_step = new_id("dream_"), 0
        elif t == EventType.SLEEP_ENDED:
            self._finish()
        elif t == EventType.TICK:
            mode = event.data(TickPayload).body.mode
            if self._busy or mode not in (Mode.ASLEEP, Mode.DREAMING):
                return
            if self._started is None:  # asleep at boot or entered without the start event
                self._started = self.now()
            self._busy = True
            self.bus.spawn(self._work(mode), name="sleep-work")

    async def _work(self, mode: Mode) -> None:
        try:
            if mode == Mode.ASLEEP:
                await self.consolidate_step()
            else:
                await self.dream()
        finally:
            self._busy = False

    # ------------------------------------------------------------------ NREM-like
    async def consolidate_step(self) -> None:
        batch = await self.ask_one("memory.replay_batch", {"n": self.cfg.replay_batch}, default=None)
        mems = [MemoryRef.model_validate(m) for m in (batch or {}).get("memories", [])]
        if not mems:
            await self._maintenance()
            return
        self.emit(EventType.MEMORY_REPLAYED, MemoryReplayedPayload(memories=mems),
                  summary=f"replaying {len(mems)} memories: {truncate(mems[0].content, 80)}",
                  modality=Modality.MEMORY, nominated=True, salience=0.3)
        self.stats.replayed += len(mems)
        sim = batch.get("similarity") or []
        thr = max(self.cfg.cluster_similarity, self.ctx.embedder.relevance_floor + 0.25)
        clusters: list[list[int]] = []
        for i in range(len(mems)):
            for c in clusters:
                if all(sim[i][j] >= thr for j in c):
                    c.append(i)
                    break
            else:
                clusters.append([i])
        for c in clusters:
            group = [mems[i] for i in c]
            if len(group) < 2 and group[0].importance < 0.7:
                continue
            contents = [m.content for m in group]
            abstraction = rule_abstract(contents)  # regularities in its own memories, nothing invented
            for fact in abstraction.facts[:3]:
                await self.ask_one("memory.store", {
                    "kind": "semantic", "content": fact, "source": "sleep consolidation",
                    "importance": round(sum(m.importance for m in group) / len(group), 3),
                    "confidence": min(0.9, 0.5 + 0.1 * len(group)),
                    "context": {"derived_from": [m.id for m in group]}})
                self.stats.abstractions.append(fact)
        ids = [m.id for m in mems]
        self.stats.strengthened += await self.ask_one("memory.strengthen", {"ids": ids, "amount": 0.1}, default=0) or 0
        await self.ask_one("memory.mark_consolidated", {"ids": ids}, default=0)

    async def _maintenance(self) -> None:
        if self._maintenance_done:
            return
        self._maintenance_done = True
        await self.ask_one("memory.fade", {}, default=0)  # faded episodes keep only their gist
        self.stats.pruned += await self.ask_one("memory.prune", {
            "threshold": self.cfg.prune_importance, "min_age_s": self.cfg.prune_min_age_s}, default=0) or 0
        await self.ask_one("self.reflect", {}, default=None)
        await self.ask_one("world.maintain", {}, default=None)

    # ------------------------------------------------------------------ REM-like
    async def dream(self) -> None:
        if self.dream_id is None:
            self.dream_id, self.dream_step = new_id("dream_"), 0
        frags = await self.ask_one("memory.sample", {"n": self.cfg.dream_fragments * 3}, default=[]) or []
        contents = [f["content"] for f in frags if not f["content"].startswith(
            ("I thought:", "I decided", "I was surprised", "I adopted", "I started up", "I slept", "I tried"))]
        contents = contents[: self.cfg.dream_fragments]
        out = await self.ask_one("imagination.dream", {"fragments": contents}, default=None)
        narrative = (out or {}).get("narrative") or (
            f"A dream-like blend of: {'; '.join(truncate(c, 60) for c in contents)}" if contents
            else "A formless dream-like state with no content.")
        self.emit(EventType.DREAM_CONTENT,
                  DreamPayload(dream_id=self.dream_id, narrative=narrative, fragments=[f["id"] for f in frags],
                               step=self.dream_step),
                  summary=f"(dream-like simulation) {truncate(narrative, 140)}", modality=Modality.IMAGINATION,
                  nominated=True, salience=0.55, confidence=0.2)
        self.dream_step += 1
        self.stats.dreams += 1
        self.dreams = ([{"t": self.now(), "dream_id": self.dream_id, "narrative": narrative}] + self.dreams)[:20]

    def _finish(self) -> None:
        if self._started is None:
            return
        self.stats.duration_s = round(self.now() - self._started, 1)
        s = self.stats
        self.history = ([{"t": self.now(), **s.model_dump()}] + self.history)[:10]
        self.emit(EventType.MEMORY_CONSOLIDATED, s,
                  summary=(f"slept {s.duration_s:.0f}s: replayed {s.replayed}, abstracted {len(s.abstractions)}, "
                           f"pruned {s.pruned}, {s.dreams} dream-like simulations"),
                  modality=Modality.MEMORY, nominated=True, salience=0.4)
        self._started = None
        self.dream_id = None

    def snapshot(self) -> dict:
        return {"current": self.stats.model_dump(), "dreams": self.dreams[:6], "history": self.history[:5],
                "busy": self._busy}

    def trace_state(self) -> dict:
        return {"replayed": self.stats.replayed, "dreams": self.stats.dreams}
