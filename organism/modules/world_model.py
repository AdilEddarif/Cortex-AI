"""World model: persistent, uncertain representation of the environment.

Entities (people, objects, places, sounds, concepts) carry *beliefs*, each with a value,
confidence, source and observation time. Confidence of perishable beliefs (presence of an
object, someone speaking) decays with time since the last observation, which gives object
permanence with growing uncertainty. Model outputs are never stored as bare facts.
New observations that contradict confident beliefs raise WORLD_CONFLICT.
"""
from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, Field

from ..core.events import (
    Event, EventType, PerceptPayload, UtterancePayload, WorldConflictPayload, WorldPayload,
)
from ..core.module import CognitiveModule
from ..core.util import humanize_duration

SINGLE_VALUED = {"name", "lives_in", "works_as", "age", "location"}
PERSON_LABELS = {"person", "man", "woman", "child", "people"}


class Belief(BaseModel):
    value: Any
    confidence: float
    source: str
    observed_at: float
    tau: float | None = None  # seconds; None = stable belief

    def effective(self, now: float) -> float:
        if self.tau is None:
            return self.confidence
        return self.confidence * math.exp(-max(0.0, now - self.observed_at) / self.tau)


class Entity(BaseModel):
    id: str
    kind: str
    name: str
    modality: str | None = None
    attributes: dict[str, Belief] = Field(default_factory=dict)
    location: str | None = None
    first_seen: float = 0.0
    last_seen: float = 0.0
    observations: int = 0


class WorldModel(CognitiveModule):
    name = "world_model"
    subscriptions = (EventType.PERCEPTION, EventType.UTTERANCE_UNDERSTOOD)

    def __init__(self, ctx):
        super().__init__(ctx)
        self.entities: dict[str, Entity] = {}
        self.relations: list[dict] = []
        self.changes: list[dict] = []

    async def start(self) -> None:
        if self.ctx.ltm_enabled:
            for d in self.ctx.store.load_entities():
                e = Entity.model_validate(d)
                self.entities[e.id] = e
        now = self.now()
        here = self._entity("place:here", "place", self.settings.world.location_name, now)
        here.attributes["description"] = Belief(value=self.settings.world.location_name, confidence=0.6,
                                                source="configuration", observed_at=now)
        me = self._entity("self", "self", self.settings.identity.name, now)
        me.location = "place:here"
        for topic, fn in {"world.describe": self.describe, "world.query": self.query,
                          "world.maintain": self.maintain}.items():
            self.respond(topic, fn)

    async def stop(self) -> None:
        for e in self.entities.values():
            self._save(e)

    def _entity(self, eid: str, kind: str, name: str, now: float, modality: str | None = None) -> Entity:
        e = self.entities.get(eid)
        if e is None:
            e = Entity(id=eid, kind=kind, name=name, modality=modality, first_seen=now, last_seen=now,
                       location="place:here" if kind in ("object", "person", "sound") else None)
            self.entities[eid] = e
        return e

    def _save(self, e: Entity) -> None:
        if self.ctx.ltm_enabled:
            self.ctx.store.put_entity(e.id, e.model_dump(mode="json"), self.now())

    def _believe(self, e: Entity, attr: str, value: Any, conf: float, source: str, tau: float | None = None) -> None:
        now = self.now()
        old = e.attributes.get(attr)
        if (old is not None and attr in SINGLE_VALUED and str(old.value).lower() != str(value).lower()
                and old.effective(now) >= 0.6):
            self.emit(EventType.WORLD_CONFLICT,
                      WorldConflictPayload(entity_id=e.id, attribute=attr, believed=old.value,
                                           believed_confidence=round(old.effective(now), 3), observed=value,
                                           observed_confidence=conf),
                      summary=f"conflict about {e.name}'s {attr}: I believed '{old.value}', now '{value}'",
                      nominated=True, salience=0.55, confidence=conf)
            conf *= 0.8  # remain less certain after a contradiction
        e.attributes[attr] = Belief(value=value, confidence=round(conf, 3), source=source, observed_at=now, tau=tau)
        if attr == "name" and e.kind == "person":
            e.name = str(value)

    def _changed(self, e: Entity, changes: dict) -> None:
        e.last_seen = self.now()
        e.observations += 1
        self._save(e)
        self.changes = ([{"t": self.now(), "entity": e.name, "changes": list(changes)}] + self.changes)[:30]
        self.emit(EventType.WORLD_UPDATED, WorldPayload(entity_id=e.id, changes=changes),
                  summary=f"world: {e.name} updated ({', '.join(changes)[:80]})")

    # ------------------------------------------------------------------ updates
    async def handle(self, event: Event) -> None:
        now = self.now()
        if event.type == EventType.PERCEPTION:
            p = event.data(PerceptPayload)
            if p.self_generated:
                return
            if p.vision is not None:
                labels: dict[str, list[float]] = {}
                for o in p.vision.objects:
                    labels.setdefault(o.label, []).append(o.confidence)
                for label, confs in labels.items():
                    kind = "person" if label in PERSON_LABELS else "object"
                    e = self._entity(f"obj:{label}", kind, label, now, "vision")
                    self._believe(e, "present", True, max(confs), "vision", tau=120.0)
                    self._believe(e, "count", len(confs), max(confs), "vision", tau=120.0)
                    self._changed(e, {"present": True, "count": len(confs)})
                for e in [x for x in self.entities.values() if x.modality == "vision" and x.name not in labels]:
                    b = e.attributes.get("present")
                    if b is not None and b.value is True:
                        self._believe(e, "present", False, 0.5, "vision: not seen in the latest view", tau=120.0)
                        self._changed(e, {"present": False})
                if p.vision.spatial_relations:
                    self.relations = [dict(r.model_dump(), observed_at=now, source="vision")
                                      for r in p.vision.spatial_relations]
                if p.vision.scene:
                    here = self.entities["place:here"]
                    self._believe(here, "appearance", p.vision.scene, 0.6, "vision-language model", tau=600.0)
                    self._changed(here, {"appearance": p.vision.scene})
            elif p.audio is not None and p.audio.kind == "sound":
                label = p.audio.sound_label or "a sound"
                e = self._entity(f"sound:{label}", "sound", label, now, "audio")
                self._believe(e, "heard", True, event.confidence, "audio", tau=60.0)
                self._believe(e, "loudness_db", round(p.audio.loudness_db, 1), 0.9, "audio", tau=60.0)
                self._changed(e, {"heard": True})
            elif p.text is not None or (p.audio is not None and p.audio.kind == "speech"):
                speaker = (p.text.speaker if p.text else p.audio.speaker) or "someone"
                e = self._entity(f"person:{speaker.lower()}", "person", speaker, now, p.modality.value)
                self._believe(e, "activity", "talking to me", 0.9, p.modality.value, tau=300.0)
                self._believe(e, "present", True, 0.8, p.modality.value, tau=300.0)
                self._changed(e, {"activity": "talking to me"})
        elif event.type == EventType.UTTERANCE_UNDERSTOOD:
            a = event.data(UtterancePayload).analysis
            speaker_id = f"person:{a.speaker.lower()}"
            for f in a.facts:
                if f.subject.lower() in (a.speaker.lower(), "user"):
                    e = self._entity(speaker_id, "person", a.speaker, now)
                    attr = f.relation if f.relation in SINGLE_VALUED or f.relation.startswith("favorite_") \
                        else f"{f.relation}:{f.object.lower()}"
                    self._believe(e, attr, f.object, f.confidence, f"told by {a.speaker}")
                else:
                    e = self._entity(f"concept:{f.subject.lower()}", "concept", f.subject, now)
                    self._believe(e, f.relation, f.object, f.confidence * 0.9, f"told by {a.speaker}")
                self._changed(e, {f.relation: f.object})
            if a.sentiment:
                e = self._entity(speaker_id, "person", a.speaker, now)
                mood = "positive" if a.sentiment > 0.2 else "negative" if a.sentiment < -0.2 else "neutral"
                # A guess from word choice only: explicitly low confidence.
                self._believe(e, "estimated_emotion", mood, 0.35, "inferred from word choice", tau=300.0)

    # ------------------------------------------------------------------ queries
    def _view(self, e: Entity, now: float) -> dict:
        pres = e.attributes.get("present") or e.attributes.get("heard")
        presence = pres.effective(now) if pres is not None and pres.value is True else 0.0
        return {
            "id": e.id, "kind": e.kind, "name": e.name, "modality": e.modality, "location": e.location,
            "age_s": round(now - e.last_seen, 1), "observations": e.observations,
            "presence_confidence": round(presence, 3),
            "attributes": {k: {"value": b.value, "confidence": round(b.effective(now), 3), "source": b.source}
                           for k, b in e.attributes.items()},
        }

    def statements(self, now: float) -> list[str]:
        out = []
        for e in self.entities.values():
            if e.kind == "self":
                out.append(f"I (the observer) am located at {self.settings.world.location_name}.")
                continue
            if e.kind == "place":
                continue
            v = self._view(e, now)
            ago = humanize_duration(v["age_s"])
            if e.kind in ("object", "person") and e.modality == "vision":
                c = v["presence_confidence"]
                if c >= 0.05:
                    out.append(f"I believe there is a {e.name} here (confidence {c:.2f}; last seen {ago} ago).")
            elif e.kind == "sound" and v["presence_confidence"] >= 0.05:
                out.append(f"I heard {e.name} {ago} ago.")
            elif e.kind == "person":
                for k, a in v["attributes"].items():
                    if k in ("present",) or a["confidence"] < 0.1:
                        continue
                    label = k.split(":")[0].replace("_", " ")
                    out.append(f"{e.name}: {label} = {a['value']} (confidence {a['confidence']:.2f}, {a['source']}).")
            elif e.kind == "concept":
                for k, a in v["attributes"].items():
                    out.append(f"{e.name} {k} {a['value']} (confidence {a['confidence']:.2f}).")
        for r in self.relations[:6]:
            out.append(f"The {r['subject']} is {r['relation'].replace('_', ' ')} the {r['object']} (vision).")
        return out

    async def describe(self, _q: dict) -> dict:
        now = self.now()
        return {"entities": [self._view(e, now) for e in self.entities.values() if e.kind not in ("self",)],
                "relations": self.relations, "statements": self.statements(now)}

    async def query(self, q: dict) -> dict | None:
        now = self.now()
        e = self.entities.get(q.get("id", ""))
        if e is None and q.get("name"):
            e = next((x for x in self.entities.values() if x.name.lower() == str(q["name"]).lower()), None)
        return self._view(e, now) if e else None

    async def maintain(self, _q: dict) -> int:
        now = self.now()
        stale = [eid for eid, e in self.entities.items() if e.kind in ("object", "sound")
                 and now - e.last_seen > 86400
                 and all(b.effective(now) < 0.02 for b in e.attributes.values())]
        for eid in stale:
            self.entities.pop(eid)
            if self.ctx.ltm_enabled:
                self.ctx.store.delete_entity(eid)
        return len(stale)

    def snapshot(self) -> dict:
        now = self.now()
        return {"entities": [self._view(e, now) for e in self.entities.values()],
                "relations": self.relations, "statements": self.statements(now)[:20], "changes": self.changes[:10]}

    def trace_state(self) -> dict:
        return {"n": len(self.entities)}
