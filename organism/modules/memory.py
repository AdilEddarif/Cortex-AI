"""Long-term memory (hippocampal-cortical system).

Kinds
  episodic         events the organism experienced ("user said to me: ...")
  semantic         facts and abstractions ("The user's name is Ada.")
  procedural       action-outcome statistics / skills ("Procedure 'speak' succeeded 12/12")
  autobiographical facts about the organism's own history ("I was restarted after 2 hours")
  dream            dream-like simulations - stored, but tagged so they are never mistaken
                   for real experience (reality monitoring)
  imagined         deliberate imaginings

Encoding happens when attended items are broadcast; importance depends on salience,
novelty, prediction error, goal relevance, social relevance and the current emotional
intensity (emotion literally changes what gets remembered). Retrieval is cue-driven
(automatic pattern completion on each broadcast) or explicit (``memory.recall``) and scores
memories by relevance, recency and importance x strength.

Temporal memory (``temporal_memory.py``): memories follow a forgetting curve whose stability
grows with importance, emotion and spaced recall; faded memories need a strong cue; during sleep
faded episodes are reduced to their gist; and experience is segmented into episodes (at pauses,
sleep and big surprises) that can be recalled by time ("yesterday evening").
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from ..core.events import (
    ActionPayload, ActionType, ConsolidationPayload, DreamPayload, EmotionPayload, Event, EventType,
    MemoryRef, MemoryRetrievedPayload, MemoryStoredPayload, Modality, Mode, ModePayload, SimulationPayload,
    SpeechPayload, SystemPayload, UtterancePayload, WorkspaceItem, WorkspacePayload,
)
from ..core.module import CognitiveModule
from ..core.util import clamp, fmt_clock, half_life_decay, humanize_duration, new_id, truncate
from .temporal_memory import (
    Timeline, gist, initial_stability, retention, stability_after_recall, time_label, time_reference,
)

REAL_KINDS = ("episodic", "semantic", "procedural", "autobiographical")
ALL_KINDS = REAL_KINDS + ("dream", "imagined")
NO_ENCODE_TYPES = {
    EventType.MEMORY_RETRIEVED, EventType.MEMORY_REPLAYED, EventType.DREAM_CONTENT, EventType.SPEECH_GENERATED,
    EventType.ACTION_EXECUTED, EventType.ACTION_REJECTED, EventType.SYSTEM_BOOT, EventType.SIMULATION_RESULT,
    EventType.MEMORY_CONSOLIDATED, EventType.EMOTION_CHANGED, EventType.WM_UPDATED, EventType.GOAL_UPDATED,
}
CUE_TYPES = {
    EventType.UTTERANCE_UNDERSTOOD, EventType.PERCEPTION, EventType.PREDICTION_ERROR,
    EventType.GOAL_CREATED, EventType.WORLD_CONFLICT,
}


@dataclass
class MemoryRecord:
    id: str
    kind: str
    content: str
    ts: float
    importance: float = 0.5
    confidence: float = 0.8
    source: str = ""
    context: dict = field(default_factory=dict)
    emotion: dict = field(default_factory=dict)
    episode_id: str | None = None
    access_count: int = 0
    last_access: float | None = None
    consolidated: int = 0
    strength: float = 1.0
    embedding: np.ndarray | None = None
    embed_model: str | None = None
    tags: list = field(default_factory=list)

    def ref(self, score: float = 0.0) -> MemoryRef:
        return MemoryRef(id=self.id, kind=self.kind, content=self.content, score=round(score, 4),
                         timestamp=self.ts, importance=round(self.importance, 3),
                         confidence=round(self.confidence, 3), source=self.source, episode_id=self.episode_id)

    def row(self) -> dict:
        return {k: getattr(self, k) for k in (
            "id", "kind", "content", "ts", "importance", "confidence", "source", "context", "emotion",
            "episode_id", "access_count", "last_access", "consolidated", "strength", "embedding",
            "embed_model", "tags")}


class MemorySystem:
    """Storage + vector index. No bus dependency, so it can be unit-tested directly."""

    def __init__(self, store, embedder, recency_half_life_s: float = 86400.0, persist: bool = True,
                 forgetting: bool = False, stability_base_s: float = 21600.0, access_floor: float = 0.05):
        self.store = store
        self.embedder = embedder
        self.recency_half_life_s = recency_half_life_s
        self.persist = persist
        self.forgetting = forgetting
        self.stability_base_s = stability_base_s
        self.access_floor = access_floor
        self.records: dict[str, MemoryRecord] = {}
        self._matrix: np.ndarray | None = None
        self._ids: list[str] = []

    async def load(self) -> int:
        stale = []
        for row in self.store.load_memories():
            rec = MemoryRecord(**row)
            self.records[rec.id] = rec
            if rec.embed_model != self.embedder.name or rec.embedding is None:
                stale.append(rec)
        for i in range(0, len(stale), 64):  # re-embed memories from another embedding space
            batch = stale[i:i + 64]
            vecs = await self.embedder.embed([r.content for r in batch])
            for r, v in zip(batch, vecs):
                r.embedding, r.embed_model = v, self.embedder.name
                self.store.update_memory(r.id, embedding=v, embed_model=r.embed_model)
        self._matrix = None
        return len(self.records)

    def _index(self) -> tuple[np.ndarray, list[str]]:
        if self._matrix is None:
            self._ids = [r.id for r in self.records.values() if r.embedding is not None
                         and r.embed_model == self.embedder.name]
            self._matrix = (np.stack([self.records[i].embedding for i in self._ids])
                            if self._ids else np.zeros((0, max(self.embedder.dims, 1)), np.float32))
        return self._matrix, self._ids

    async def add(self, kind: str, content: str, ts: float, *, importance: float = 0.5, confidence: float = 0.8,
                  source: str = "", context: dict | None = None, emotion: dict | None = None,
                  episode_id: str | None = None, tags: list | None = None, strength: float = 1.0) -> MemoryRecord:
        vec = await self.embedder.embed_one(content)
        rec = MemoryRecord(id=new_id("m_"), kind=kind, content=content, ts=ts, importance=clamp(importance),
                           confidence=clamp(confidence), source=source, context=context or {},
                           emotion=emotion or {}, episode_id=episode_id, strength=strength,
                           embedding=vec, embed_model=self.embedder.name, tags=tags or [])
        self.records[rec.id] = rec
        self._matrix = None
        if self.persist:
            self.store.put_memory(rec.row())
        return rec

    # forgetting curve ----------------------------------------------------------
    def stability(self, rec: MemoryRecord) -> float:
        s = rec.context.get("stability_s")
        return float(s) if s else initial_stability(rec.kind, rec.importance, 0.0, self.stability_base_s)

    def retention(self, rec: MemoryRecord, now: float) -> float:
        if not self.forgetting:
            return 1.0
        return retention(now - max(rec.ts, rec.last_access or 0.0), self.stability(rec))

    def effective_strength(self, rec: MemoryRecord, now: float) -> float:
        return rec.strength * self.retention(rec, now)

    def save(self, rec: MemoryRecord, **fields: Any) -> None:
        for k, v in fields.items():
            setattr(rec, k, v)
        if self.persist:
            self.store.update_memory(rec.id, **fields)

    async def search(self, query: str, *, kinds: Iterable[str] | None = None, k: int = 5, now: float,
                     exclude_ids: set[str] | None = None, min_relevance: float | None = None,
                     touch: bool = True) -> list[tuple[MemoryRecord, float, float]]:
        mat, ids = self._index()
        if not ids:
            return []
        q = await self.embedder.embed_one(query)
        if mat.shape[1] != q.shape[0]:
            return []
        sims = mat @ q
        kinds = set(kinds or REAL_KINDS)
        floor = self.embedder.relevance_floor if min_relevance is None else min_relevance
        out = []
        for idx in np.argsort(-sims)[: max(k * 8, 32)]:
            rel = float(sims[idx])
            if rel < floor:
                break
            rec = self.records[ids[idx]]
            if rec.kind not in kinds or (exclude_ids and rec.id in exclude_ids):
                continue
            rel_n = (rel - floor) / max(1e-6, 1.0 - floor)
            if self.forgetting:
                recency = self.retention(rec, now)
                eff = rec.strength * recency
                if eff < self.access_floor and rel_n < 0.8:
                    continue  # faded: it would take a much stronger cue to bring this back
            else:
                recency = half_life_decay(now - max(rec.ts, rec.last_access or 0.0), self.recency_half_life_s)
                eff = rec.strength
            score = 0.6 * rel_n + 0.15 * recency + 0.25 * rec.importance * min(1.0, eff / 1.5)
            out.append((rec, clamp(score), rel))
        out.sort(key=lambda x: -x[1])
        out = out[:k]
        if touch:
            for rec, _s, _r in out:  # retrieval practice strengthens memories (more when spaced)
                fields: dict[str, Any] = {"access_count": rec.access_count + 1, "last_access": now,
                                          "strength": min(3.0, rec.strength + 0.05)}
                if self.forgetting:
                    fields["context"] = {**rec.context, "stability_s": round(stability_after_recall(
                        self.stability(rec), self.retention(rec, now)), 1)}
                self.save(rec, **fields)
        return out

    def recent(self, kinds: Iterable[str] | None = None, n: int = 10, now: float | None = None) -> list[MemoryRecord]:
        kinds = set(kinds or REAL_KINDS)
        recs = [r for r in self.records.values() if r.kind in kinds
                and (now is None or self.effective_strength(r, now) >= self.access_floor)]
        return sorted(recs, key=lambda r: -r.ts)[:n]

    def similarity_matrix(self, recs: list[MemoryRecord]) -> list[list[float]]:
        vs = [r.embedding for r in recs]
        if not vs or any(v is None for v in vs):
            return [[0.0] * len(recs) for _ in recs]
        m = np.stack(vs)
        return np.round(m @ m.T, 4).tolist()

    def forget(self, rec: MemoryRecord) -> None:
        self.records.pop(rec.id, None)
        self._matrix = None
        if self.persist:
            self.store.delete_memory(rec.id)

    def counts(self) -> dict[str, int]:
        c: dict[str, int] = {k: 0 for k in ALL_KINDS}
        for r in self.records.values():
            c[r.kind] = c.get(r.kind, 0) + 1
        return c


class Memory(CognitiveModule):
    name = "memory"
    subscriptions = (
        EventType.WORKSPACE_UPDATED, EventType.SPEECH_GENERATED, EventType.ACTION_EXECUTED,
        EventType.ACTION_REJECTED, EventType.EMOTION_CHANGED, EventType.SYSTEM_BOOT,
        EventType.DREAM_CONTENT, EventType.SIMULATION_RESULT, EventType.MEMORY_CONSOLIDATED,
        EventType.SELF_STATE_CHANGED, EventType.ACTION_APPROVED, EventType.MODE_CHANGED,
    )

    def __init__(self, ctx):
        super().__init__(ctx)
        self.cfg = self.settings.memory
        self.system = MemorySystem(ctx.store, ctx.embedder, self.cfg.recency_half_life_s,
                                   forgetting=self.cfg.forgetting, stability_base_s=self.cfg.stability_base_s,
                                   access_floor=self.cfg.access_floor)
        self.timeline = Timeline()
        self.asleep = False
        self.stats = {"gisted": 0, "episodes_closed": 0}
        self.emotion: dict[str, float] = {}
        self.emotion_intensity = 0.0
        self._refractory: dict[str, float] = {}
        self.recent_retrievals: list[dict] = []
        self.recent_encodings: list[dict] = []

    async def start(self) -> None:
        n = await self.system.load()
        self.log.info("loaded %d long-term memories", n)
        if self.cfg.episodes:
            self.timeline = Timeline(self.kv_load("timeline", []) or [])
        for topic, fn in {
            "memory.recall": self._recall, "memory.recent": self._recent, "memory.store": self._store,
            "memory.replay_batch": self._replay_batch, "memory.mark_consolidated": self._mark_consolidated,
            "memory.strengthen": self._strengthen, "memory.prune": self._prune, "memory.sample": self._sample,
            "memory.procedural": self._procedural, "memory.facts": self._facts, "memory.stats": self._stats,
            "memory.fade": self._fade, "memory.episodes": self._episodes,
            "executor.remember": self._executor_flag,
        }.items():
            self.respond(topic, fn)

    async def stop(self) -> None:
        self._save_timeline()

    def _save_timeline(self) -> None:
        if self.cfg.episodes:
            self.kv_save("timeline", self.timeline.to_list())

    def _boundary(self, reason: str) -> None:
        """Event segmentation: the current episode ends here."""
        if self.cfg.episodes and self.timeline.close(reason):
            self.stats["episodes_closed"] += 1
            self._save_timeline()

    # ------------------------------------------------------------------ event handling
    async def handle(self, event: Event) -> None:
        t = event.type
        now = self.now()
        if t == EventType.MODE_CHANGED:
            p = event.data(ModePayload)
            sleeping = p.current in (Mode.ASLEEP, Mode.DREAMING)
            if sleeping and not self.asleep:
                self._boundary("sleep")
            self.asleep = sleeping
        elif t == EventType.EMOTION_CHANGED:
            p = event.data(EmotionPayload)
            self.emotion, self.emotion_intensity = p.state, p.intensity
        elif t == EventType.WORKSPACE_UPDATED:
            await self._on_broadcast(event.data(WorkspacePayload))
        elif t == EventType.SPEECH_GENERATED:
            s = event.data(SpeechPayload)
            await self.encode("episodic", f'I said to {s.addressed_to}: "{s.text}"', importance=0.4,
                              source="self/speech", context={"event_id": event.id, "intention": s.intention},
                              participants=[s.addressed_to])
        elif t == EventType.ACTION_EXECUTED:
            await self._on_action(event)
        elif t == EventType.ACTION_APPROVED:
            if event.payload.get("spec", {}).get("action") == ActionType.REMEMBER.value:
                await self._execute_remember(event)
        elif t == EventType.ACTION_REJECTED:
            p = event.data(ActionPayload)
            await self.encode("autobiographical", f"I tried to {p.spec.action.value} but could not: {p.reason}",
                              importance=0.55, source="self/action", context={"event_id": event.id})
        elif t == EventType.SYSTEM_BOOT:
            info = event.data(SystemPayload).info
            down = info.get("downtime_s")
            text = f"I started up (boot #{info.get('boot_count', '?')})."
            if down:
                text += f" I had been offline for {humanize_duration(down)}."
            await self.encode("autobiographical", text, importance=0.6, source="self/system",
                              context={"event_id": event.id})
        elif t == EventType.DREAM_CONTENT:
            d = event.data(DreamPayload)
            await self.encode("dream", d.narrative, importance=0.3, confidence=0.2, source="self/dream",
                              context={"event_id": event.id, "dream_id": d.dream_id, "fragments": d.fragments})
        elif t == EventType.SIMULATION_RESULT and event.nominated:
            s = event.data(SimulationPayload)
            await self.encode("imagined", f"I imagined: {s.narrative or s.predicted_outcome}", importance=0.35,
                              confidence=0.3, source="self/imagination", context={"event_id": event.id})
        elif t == EventType.MEMORY_CONSOLIDATED:
            c = event.data(ConsolidationPayload)
            if c.duration_s > 0:
                await self.encode("autobiographical",
                                  f"I slept for {humanize_duration(c.duration_s)}: replayed {c.replayed} memories, "
                                  f"formed {len(c.abstractions)} abstractions and had {c.dreams} dream-like simulations.",
                                  importance=0.45, source="self/sleep", context={"event_id": event.id})
        elif t == EventType.SELF_STATE_CHANGED:
            lim = event.payload.get("changes", {}).get("new_limitation")
            if lim:
                await self.encode("autobiographical", f"I learned a limitation of mine: {lim}", importance=0.6,
                                  source="self/self_model", context={"event_id": event.id})
        self._refractory = {k: v for k, v in self._refractory.items() if now - v < self.cfg.refractory_s}

    async def _on_broadcast(self, p: WorkspacePayload) -> None:
        new = [i for i in p.items if i.event_id in set(p.new_item_ids)]
        for item in new:
            if item.event_type == EventType.PREDICTION_ERROR and self.cfg.episodes:
                pl = item.event.get("payload", {})
                cur = self.timeline.current
                if (float(pl.get("error", 0)) >= self.cfg.boundary_surprise and cur and cur.n >= 3
                        and pl.get("target") in ("visual_scene", "conversation")):
                    self._boundary("surprise")  # a big surprise starts a new chapter
                    self.ctx.temporal.new_episode(self.now())
            if item.event_type == EventType.MEMORY_RETRIEVED:
                ids = [m["id"] for m in item.event.get("payload", {}).get("memories", [])]
                await self._strengthen({"ids": ids, "amount": 0.1})
                continue
            routine_goal = (item.event_type == EventType.GOAL_CREATED and
                            item.event.get("payload", {}).get("goal", {}).get("origin") in ("user", "social"))
            if item.event_type not in NO_ENCODE_TYPES and not routine_goal:
                imp = self.importance(item)
                if item.event_type == EventType.UTTERANCE_UNDERSTOOD:
                    await self._learn_facts(item)
                if imp >= self.cfg.encode_threshold:
                    # The valuation system tags the moment with its own emotional charge (amygdala-like),
                    # so an emotional event is encoded strongly even before the mood has caught up.
                    charge = abs(float(item.event.get("emotional_value", 0.0))) if self.ctx.settings.enabled("emotion") else 0.0
                    await self.encode("episodic", self.episodic_text(item), importance=imp, charge=charge,
                                      confidence=float(item.event.get("confidence", 0.8)),
                                      source=item.source,
                                      context={"event_id": item.event_id, "event_type": item.event_type.value,
                                               "modality": item.modality.value},
                                      participants=self._participants(item))
            if item.event_type in CUE_TYPES or (
                    item.event_type == EventType.THOUGHT_GENERATED
                    and item.event.get("payload", {}).get("kind") != "mind_wandering"):
                await self.cue_retrieval(item)

    def importance(self, item: WorkspaceItem) -> float:
        c = item.components or {"salience": float(item.event.get("salience", 0.3)),
                                "urgency": float(item.event.get("urgency", 0.0))}
        social = 0.25 if item.event_type == EventType.UTTERANCE_UNDERSTOOD else 0.0
        if social and item.event.get("payload", {}).get("analysis", {}).get("facts"):
            social += 0.2  # informative utterances (someone told me something) matter more
        imp = (0.25 * c.get("salience", 0) + 0.2 * c.get("novelty", 0) + 0.15 * c.get("goal_relevance", 0)
               + 0.15 * c.get("emotional_relevance", 0) + 0.25 * c.get("prediction_error", 0)
               + 0.1 * c.get("urgency", 0) + 0.2 * self.emotion_intensity + social
               + 0.3 * float(item.event.get("importance", 0.0)))
        return round(clamp(0.75 * imp), 4)  # scaled so that importance rarely saturates

    @staticmethod
    def episodic_text(item: WorkspaceItem) -> str:
        p = item.event.get("payload", {})
        t = item.event_type
        if t == EventType.UTTERANCE_UNDERSTOOD:
            a = p.get("analysis", {})
            return f'{a.get("speaker", "someone")} said to me: "{a.get("text", "")}"'
        if t == EventType.PERCEPTION:
            verb = {"vision": "I saw", "audio": "I heard"}.get(p.get("modality"), "I perceived")
            desc = str(p.get("description", item.summary))
            return desc if desc.startswith(("I saw", "I heard", "I perceived", "I see", "I hear")) else f"{verb}: {desc}"
        if t == EventType.THOUGHT_GENERATED:
            return f"I thought: {p.get('content', item.summary)}"
        if t == EventType.PREDICTION_ERROR:
            return f"I was surprised: {item.summary}"
        if t == EventType.GOAL_CREATED:
            return f"I adopted the goal: {p.get('goal', {}).get('description', item.summary)}"
        return item.summary

    @staticmethod
    def _participants(item: WorkspaceItem) -> list[str]:
        a = item.event.get("payload", {}).get("analysis")
        return [a["speaker"]] if a and a.get("speaker") else []

    async def _learn_facts(self, item: WorkspaceItem) -> None:
        u = UtterancePayload.model_validate(item.event["payload"])
        for f in u.analysis.facts:
            await self._store({
                "kind": "semantic", "content": fact_sentence(f.subject, f.relation, f.object),
                "importance": 0.6, "confidence": f.confidence, "source": f"told by {u.analysis.speaker}",
                "context": {"event_id": item.event_id, "subject": f.subject, "relation": f.relation,
                            "object": f.object},
            })

    async def _on_action(self, event: Event) -> None:
        p = event.data(ActionPayload)
        outcome = str(p.result.get("outcome", "unknown"))
        await self._update_procedure(p.spec.action.value, outcome == "success", p.spec.reason)
        if p.spec.action not in (ActionType.WAIT, ActionType.SPEAK, ActionType.ASK, ActionType.REMEMBER):
            await self.encode("autobiographical",
                              f"I decided to {p.spec.action.value} because {p.spec.reason or 'it seemed useful'};"
                              f" outcome: {outcome}.", importance=0.45, source="self/action",
                              context={"event_id": event.id})

    async def _execute_remember(self, event: Event) -> None:
        """Executor for the 'remember' action: deliberately store or deliberately recall."""
        p = event.data(ActionPayload)
        content, query = p.spec.params.get("content"), p.spec.params.get("query")
        result: dict[str, Any] = {"outcome": "success"}
        if content:
            rec = await self.encode("semantic", content, importance=0.8, confidence=0.8,
                                    source="deliberately memorised", context={"event_id": event.id})
            result["stored"] = rec.id
        elif query:
            hits = await self.system.search(query, k=3, now=self.now())
            result["recalled"] = [r.content for r, _s, _rel in hits]
            if hits:
                refs = [r.ref(s) for r, s, _ in hits]
                self.emit(EventType.MEMORY_RETRIEVED, MemoryRetrievedPayload(cue=query, memories=refs),
                          summary=f"I deliberately recalled: {truncate(refs[0].content, 110)}",
                          modality=Modality.MEMORY, nominated=True, salience=0.6, confidence=refs[0].confidence)
            else:
                result["outcome"] = "failed"
                result["detail"] = "nothing relevant in memory"
        else:
            result = {"outcome": "failed", "detail": "nothing to remember"}
        self.emit(EventType.ACTION_EXECUTED,
                  ActionPayload(decision_id=p.decision_id, spec=p.spec, status="executed", result=result),
                  summary=f"Executed remember: {result.get('outcome')}", modality=Modality.MOTOR)

    async def _update_procedure(self, action: str, success: bool, reason: str) -> None:
        rec = next((r for r in self.system.records.values()
                    if r.kind == "procedural" and r.context.get("action") == action), None)
        if rec is None:
            ctx = {"action": action, "n": 0, "success": 0}
            rec = await self.system.add("procedural", f"Procedure '{action}'", self.now(), importance=0.4,
                                        confidence=0.5, source="self/learning", context=ctx)
        ctx = dict(rec.context)
        ctx["n"] = int(ctx.get("n", 0)) + 1
        ctx["success"] = int(ctx.get("success", 0)) + int(success)
        rate = ctx["success"] / ctx["n"]
        content = (f"Procedure '{action}': executed {ctx['n']} times, succeeded {ctx['success']} "
                   f"({rate:.0%}). Last used because: {truncate(reason, 80)}")
        self.system.save(rec, context=ctx, content=content, confidence=clamp(0.5 + 0.05 * ctx["n"], 0, 0.95))

    # ------------------------------------------------------------------ encoding / retrieval
    async def encode(self, kind: str, content: str, *, importance: float, confidence: float = 0.8,
                     source: str = "", context: dict | None = None, participants: list[str] | None = None,
                     charge: float = 0.0) -> MemoryRecord:
        now = self.now()
        ctx = {"time": fmt_clock(now), "place": self.settings.world.location_name,
               "participants": participants or [], "mode": None, **(context or {})}
        arousal = max(self.emotion_intensity, charge)
        strength = 0.5 + 0.5 * arousal + 0.5 * importance  # emotional/important => stronger
        ctx["stability_s"] = round(initial_stability(kind, importance, arousal, self.cfg.stability_base_s), 1)
        rec = await self.system.add(kind, content, now, importance=importance, confidence=confidence, source=source,
                                    context=ctx, emotion=dict(self.emotion),
                                    episode_id=self.ctx.temporal.episode_id or None, strength=strength)
        if self.cfg.episodes and not self.asleep and kind in REAL_KINDS and kind != "procedural":
            ep = self.timeline.observe(self.ctx.temporal.episode_id, kind, content, importance, now, participants or [])
            if ep.n % 10 == 1:
                self._save_timeline()
        self.recent_encodings = ([{"kind": kind, "content": content, "importance": round(importance, 3),
                                   "t": now}] + self.recent_encodings)[:30]
        source_event = ctx.get("event_id")
        self.emit(EventType.MEMORY_STORED, MemoryStoredPayload(memory=rec.ref()),
                  summary=f"stored {kind} memory: {truncate(content, 100)}", modality=Modality.MEMORY,
                  importance=importance,
                  caused_by=[source_event] if isinstance(source_event, str) else [])
        return rec

    async def cue_retrieval(self, item: WorkspaceItem) -> None:
        now = self.now()
        hits = await self.system.search(item.summary, k=3, now=now, exclude_ids=set(self._refractory),
                                        touch=False)
        hits = [(r, s, rel) for r, s, rel in hits
                if r.context.get("event_id") != item.event_id and now - r.ts > 1.0
                and s >= self.cfg.auto_retrieve_threshold]
        if not hits:
            return
        for r, _s, _rel in hits:
            self._refractory[r.id] = now
        refs = [r.ref(s) for r, s, _ in hits]
        top = refs[0]
        when = fmt_clock(top.timestamp)
        self.recent_retrievals = ([{"cue": item.summary, "memories": [m.content for m in refs], "t": now}]
                                  + self.recent_retrievals)[:20]
        self.emit(EventType.MEMORY_RETRIEVED, MemoryRetrievedPayload(cue=item.summary, memories=refs),
                  summary=f"I recall ({top.kind}, {when}): {truncate(top.content, 110)}",
                  modality=Modality.MEMORY, nominated=True, salience=round(0.25 + 0.45 * top.score, 3),
                  confidence=round(sum(m.confidence for m in refs) / len(refs), 3),
                  importance=top.importance)

    # ------------------------------------------------------------------ responders
    async def _recall(self, q: dict) -> list[dict]:
        kinds = q.get("kinds") or (ALL_KINDS if q.get("include_imagined") else REAL_KINDS)
        hits = await self.system.search(q.get("query", ""), kinds=kinds, k=int(q.get("k", self.cfg.retrieval_k)),
                                        now=self.now(), min_relevance=q.get("min_relevance"))
        return [{**r.ref(s).model_dump(), "relevance": round(rel, 3),
                 "time": fmt_clock(r.ts), "age_s": round(self.now() - r.ts, 1)} for r, s, rel in hits]

    async def _recent(self, q: dict) -> list[dict]:
        recs = self.system.recent(q.get("kinds"), int(q.get("n", 10)), now=self.now())
        return [{**r.ref().model_dump(), "time": fmt_clock(r.ts), "age_s": round(self.now() - r.ts, 1)}
                for r in recs]

    async def _store(self, q: dict) -> dict:
        kind = q.get("kind", "semantic")
        content = q["content"]
        if kind == "semantic":  # repetition strengthens instead of duplicating
            dup = await self.system.search(content, kinds=["semantic"], k=1, now=self.now(), touch=False,
                                           min_relevance=0.9)
            if dup:
                rec = dup[0][0]
                self.system.save(rec, confidence=clamp(rec.confidence + 0.1), strength=min(3.0, rec.strength + 0.2),
                                 importance=max(rec.importance, float(q.get("importance", 0.5))))
                return rec.ref().model_dump()
        rec = await self.encode(kind, content, importance=float(q.get("importance", 0.5)),
                                confidence=float(q.get("confidence", 0.7)), source=q.get("source", ""),
                                context=q.get("context"))
        return rec.ref().model_dump()

    async def _replay_batch(self, q: dict) -> dict:
        n = int(q.get("n", 6))
        now = self.now()
        cands = [r for r in self.system.records.values() if r.kind == "episodic" and not r.consolidated]
        cands.sort(key=lambda r: -(r.importance * r.strength + 0.3 * half_life_decay(now - r.ts, 3600)))
        batch = cands[:n]
        return {"memories": [r.ref().model_dump() for r in batch],
                "similarity": self.system.similarity_matrix(batch)}

    async def _mark_consolidated(self, q: dict) -> int:
        n = 0
        for mid in q.get("ids", []):
            rec = self.system.records.get(mid)
            if rec:
                self.system.save(rec, consolidated=1)
                n += 1
        return n

    async def _strengthen(self, q: dict) -> int:
        n = 0
        for mid in q.get("ids", []):
            rec = self.system.records.get(mid)
            if rec:
                self.system.save(rec, strength=min(3.0, rec.strength + float(q.get("amount", 0.1))))
                n += 1
        return n

    async def _prune(self, q: dict) -> int:
        """Forget weak, old, unimportant, never-used episodic/dream memories."""
        now = self.now()
        thr = float(q.get("threshold", 0.08))
        min_age = float(q.get("min_age_s", 86400))
        victims = [r for r in self.system.records.values()
                   if r.kind in ("episodic", "dream", "imagined") and now - r.ts > min_age
                   and r.importance * self.system.effective_strength(r, now) < thr and r.access_count == 0]
        for r in victims:
            self.system.forget(r)
        return len(victims)

    async def _fade(self, q: dict) -> int:
        """Sleep: faded episodes lose their details; only the gist is kept (flashbulb memories are spared)."""
        if not (self.cfg.gist and self.cfg.forgetting):
            return 0
        now = self.now()
        faded = [r for r in self.system.records.values()
                 if r.kind == "episodic" and not r.context.get("gist") and now - r.ts > self.cfg.gist_min_age_s
                 and r.importance < self.cfg.keep_detail_importance
                 and self.system.retention(r, now) < self.cfg.gist_retention]
        for r in faded:
            short = gist(r.content)
            if short == r.content:
                self.system.save(r, context={**r.context, "gist": True})
                continue
            vec = await self.system.embedder.embed_one(short)
            self.system.save(r, content=short, embedding=vec, context={**r.context, "gist": True})
            self.system._matrix = None
        self.stats["gisted"] += len(faded)
        return len(faded)

    def _episode_view(self, e, now: float) -> dict:
        return {"id": e.id, "start": e.start, "end": e.end, "when": time_label(e.start, now),
                "summary": e.summary or e.describe(), "participants": e.participants, "memories": e.n,
                "ongoing": e.end_reason is None, "ended_by": e.end_reason}

    async def _episodes(self, q: dict) -> dict:
        """Recall by time: ``text`` may hold a time reference ("yesterday evening", "before you slept")."""
        now = self.now()
        ref = time_reference(q.get("text", ""), now) if q.get("text") else None
        if not self.cfg.episodes:
            return {"window": ref, "episodes": [], "available": False}
        if ref is None:
            eps = [e for e in self.timeline.episodes if e.n][-int(q.get("n", 5)):]
        elif "anchor" in ref:
            e = {"before_sleep": self.timeline.before_last_sleep, "after_sleep": self.timeline.after_last_sleep,
                 "first": self.timeline.first}[ref["anchor"]]()
            eps = [e] if e else []
        else:
            eps = self.timeline.between(ref["start"], ref["end"])
        return {"window": ref, "available": True, "episodes": [self._episode_view(e, now) for e in eps[-3:]]}

    async def _sample(self, q: dict) -> list[dict]:
        """Weighted random fragments for dream-like recombination."""
        n = int(q.get("n", 3))
        pool = [r for r in self.system.records.values() if r.kind in ("episodic", "semantic", "autobiographical")]
        if not pool:
            return []
        rng: random.Random = self.ctx.rng
        weights = [0.1 + r.importance * r.strength + sum(abs(v) for v in r.emotion.values()) * 0.1 for r in pool]
        picks: list[MemoryRecord] = []
        for _ in range(min(n, len(pool))):
            r = rng.choices(pool, weights=weights, k=1)[0]
            if r not in picks:
                picks.append(r)
        return [r.ref().model_dump() for r in picks]

    async def _procedural(self, q: dict) -> dict:
        out = {}
        for r in self.system.records.values():
            if r.kind == "procedural":
                n = int(r.context.get("n", 0))
                out[r.context.get("action")] = {"n": n, "success_rate": (r.context.get("success", 0) / n) if n else None}
        return out

    async def _facts(self, q: dict) -> list[dict]:
        subject = (q.get("subject") or "").lower()
        out = []
        for r in self.system.records.values():
            if r.kind != "semantic":
                continue
            if subject and r.context.get("subject", "").lower() != subject and subject not in r.content.lower():
                continue
            out.append({**r.ref().model_dump(), **{k: r.context.get(k) for k in ("subject", "relation", "object")}})
        return sorted(out, key=lambda d: -d["timestamp"])

    async def _stats(self, _q: dict) -> dict:
        return self.system.counts()

    async def _executor_flag(self, _q: dict) -> bool:
        return True

    # ------------------------------------------------------------------ observability
    def snapshot(self) -> dict:
        recent = self.system.recent(ALL_KINDS, 12)
        return {
            "counts": self.system.counts(),
            "embedder": self.ctx.embedder.name,
            "recent": [{"kind": r.kind, "content": r.content, "time": fmt_clock(r.ts),
                        "importance": round(r.importance, 2), "strength": round(r.strength, 2),
                        "retention": round(self.system.retention(r, self.now()), 2), "gist": bool(r.context.get("gist")),
                        "consolidated": bool(r.consolidated)} for r in recent],
            "retrievals": self.recent_retrievals[:8],
            "episodes": [self._episode_view(e, self.now()) for e in self.timeline.episodes[-6:]][::-1],
            "temporal": dict(self.stats),
        }

    def trace_state(self) -> dict:
        return {"n": len(self.system.records)}


def fact_sentence(subject: str, relation: str, obj: str) -> str:
    subj = "The user" if subject.lower() in ("user", "the user") else subject
    poss = "The user's" if subject.lower() in ("user", "the user") else f"{subject}'s"
    rel = relation.replace("_", " ")
    if relation in ("name", "age", "job", "location") or relation.startswith(("favorite_", "favourite_")):
        return f"{poss} {rel} is {obj}."
    if relation in ("likes", "dislikes", "loves", "hates", "wants", "owns", "has", "lives_in", "works_as"):
        return f"{subj} {rel} {obj}."
    if relation == "is":
        return f"{subj} is {obj}."
    return f"{subj} {rel} {obj}."

