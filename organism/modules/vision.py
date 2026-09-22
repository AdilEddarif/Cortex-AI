"""Visual system.

    camera / uploaded image -> change detection -> object detector (YOLO) -> spatial relations
                             -> optional scene description (vision-language model)
                             -> structured percept -> attention

Raw pixels never leave this module; only a structured percept is published (and frames are
not stored unless ``vision.keep_frames``). Processing is skipped when the scene has not
changed (a crude saliency/efficiency mechanism), unless the organism deliberately ``look``s.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

import numpy as np

from ..core.events import (
    ActionPayload, ActionType, DetectedObject, Event, EventType, Modality, PerceptPayload, SpatialRelation,
    VisionData,
)
from ..core.module import CognitiveModule
from ..core.util import clamp, new_id

VLM_PROMPT = "Describe this scene in one short sentence for a robot's perception system. Mention people and salient objects."


def spatial_relations(objs: list[DetectedObject], limit: int = 5) -> list[SpatialRelation]:
    rels: list[SpatialRelation] = []
    boxed = [o for o in objs if o.bbox][:limit]
    for i, a in enumerate(boxed):
        for b in boxed[i + 1:]:
            ax, ay = (a.bbox[0] + a.bbox[2]) / 2, (a.bbox[1] + a.bbox[3]) / 2
            bx, by = (b.bbox[0] + b.bbox[2]) / 2, (b.bbox[1] + b.bbox[3]) / 2
            conf = round(min(a.confidence, b.confidence), 3)
            if abs(ax - bx) > 0.15:
                rels.append(SpatialRelation(subject=a.label, relation="left_of" if ax < bx else "right_of",
                                            object=b.label, confidence=conf))
            if abs(ay - by) > 0.15:
                rels.append(SpatialRelation(subject=a.label, relation="above" if ay < by else "below",
                                            object=b.label, confidence=conf))
            if ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 < 0.2:
                rels.append(SpatialRelation(subject=a.label, relation="near", object=b.label, confidence=conf))
    return rels[:12]


def describe(objs: list[DetectedObject], scene: str | None) -> str:
    if scene:
        base = scene
    elif objs:
        counts: dict[str, int] = {}
        for o in objs:
            counts[o.label] = counts.get(o.label, 0) + 1
        parts = [f"{n} {lbl}s" if n > 1 else f"a {lbl}" for lbl, n in counts.items()]
        base = "I see " + (", ".join(parts[:-1]) + " and " + parts[-1] if len(parts) > 1 else parts[0])
    else:
        base = "I see a scene with no recognisable objects"
    return base


class YoloDetector:
    def __init__(self, cfg, models_dir: Path | None = None):
        self.cfg = cfg
        self.models_dir = models_dir
        self.model = None
        self.name = None

    def load(self) -> None:
        from ultralytics import YOLO  # optional dependency
        for weights in (self.cfg.yolo_model, self.cfg.yolo_fallback_model):
            path = Path(weights)
            if self.models_dir is not None and path.name == weights:
                self.models_dir.mkdir(parents=True, exist_ok=True)
                path = self.models_dir / weights   # weights live in data/models (downloaded on first use)
            try:
                self.model = YOLO(str(path))
                self.name = path.name
                return
            except Exception:
                continue
        raise RuntimeError("no YOLO weights could be loaded")

    def detect(self, frame_bgr: np.ndarray) -> list[DetectedObject]:
        if self.model is None:
            self.load()
        h, w = frame_bgr.shape[:2]
        res = self.model.predict(frame_bgr, verbose=False, conf=self.cfg.min_confidence)[0]
        out = []
        for box in res.boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            out.append(DetectedObject(label=res.names[int(box.cls[0])], confidence=round(float(box.conf[0]), 3),
                                      bbox=(round(x1 / w, 3), round(y1 / h, 3), round(x2 / w, 3), round(y2 / h, 3))))
        return out


class Vision(CognitiveModule):
    name = "vision"
    subscriptions = (EventType.ACTION_APPROVED,)

    def __init__(self, ctx):
        super().__init__(ctx)
        self.cfg = self.settings.vision
        self.detector = (YoloDetector(self.cfg, Path(self.settings.data_dir) / "models")
                         if self.cfg.detector == "yolo" else None)
        self._detector_failed = False
        self._last_small: np.ndarray | None = None
        self._last_frame: np.ndarray | None = None
        self._last_vlm = 0.0
        self._task: asyncio.Task | None = None
        self._cap = None
        self.last_percept: dict | None = None
        self.frames = 0
        # Continuous tracking (updated on every streamed frame, used for gaze) vs. percepts
        # (published only when the stable interpretation of the scene changes).
        self.tracking: dict | None = None
        self._candidate_labels: frozenset | None = None
        self._published_labels: frozenset | None = None

    async def start(self) -> None:
        if self.cfg.enabled and not self.ctx.clock.virtual:
            self._task = asyncio.create_task(self._camera_loop(), name="camera")
        self.respond("executor.look", _true)

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
        if self._cap is not None:
            self._cap.release()

    async def _camera_loop(self) -> None:
        import cv2
        self._cap = cv2.VideoCapture(self.cfg.camera_index)
        if not self._cap.isOpened():
            self.log.warning("camera %s could not be opened", self.cfg.camera_index)
            return
        while True:
            ok, frame = await asyncio.to_thread(self._cap.read)
            if ok:
                await self.process(frame, sensor=f"camera{self.cfg.camera_index}")
            await asyncio.sleep(1.0 / max(0.1, self.cfg.fps))

    # ------------------------------------------------------------------ input
    async def ingest_image(self, data: bytes, sensor: str = "uploaded image", force: bool = True,
                           stream: bool = False) -> Event | None:
        import cv2
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("could not decode image")
        if stream:
            return await self.process_stream(frame, sensor=sensor)
        return await self.process(frame, sensor=sensor, force=force)

    async def process_stream(self, frame: np.ndarray, sensor: str) -> Event | None:
        """Live video: detect on every frame (tracking for gaze), but publish a percept only when the
        set of seen objects changes and stays changed for two consecutive frames (hysteresis), so that
        detector flicker does not become a stream of surprises."""
        change = self._change(frame)
        self._last_frame = frame
        self.frames += 1
        objs: list[DetectedObject] = []
        if self.detector is not None and not self._detector_failed:
            try:
                objs = await asyncio.to_thread(self.detector.detect, frame)
            except Exception as exc:
                self._detector_failed = True
                self.log.warning("object detector unavailable (%r); continuing without it", exc)
        self.tracking = {"t": self.now(), "objects": [o.model_dump() for o in objs],
                         "frame_size": (int(frame.shape[1]), int(frame.shape[0])), "change": round(change, 3)}
        labels = frozenset(o.label for o in objs)
        stable = labels == self._candidate_labels
        self._candidate_labels = labels
        if not stable or labels == self._published_labels:
            return None
        self._published_labels = labels
        scene = await self._vlm(frame) if self.cfg.describe_with_vlm else None
        data = VisionData(objects=objs, people=sum(1 for o in objs if o.label == "person"), scene=scene,
                          spatial_relations=spatial_relations(objs), change_score=round(max(change, 0.1), 3),
                          frame_size=(int(frame.shape[1]), int(frame.shape[0])))
        return self.publish_percept(data, sensor)

    def _change(self, frame: np.ndarray) -> float:
        import cv2
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 48)).astype(np.float32) / 255.0
        prev, self._last_small = self._last_small, small
        return 1.0 if prev is None else float(np.mean(np.abs(small - prev)))

    async def process(self, frame: np.ndarray, sensor: str, force: bool = False) -> Event | None:
        change = self._change(frame)
        if not force and change < self.cfg.change_threshold:
            return None
        self._last_frame = frame
        self.frames += 1
        objs: list[DetectedObject] = []
        if self.detector is not None and not self._detector_failed:
            try:
                objs = await asyncio.to_thread(self.detector.detect, frame)
            except Exception as exc:
                self._detector_failed = True
                self.log.warning("object detector unavailable (%r); continuing without it", exc)
        scene = await self._vlm(frame) if self.cfg.describe_with_vlm else None
        people = sum(1 for o in objs if o.label == "person")
        data = VisionData(objects=objs, people=people, scene=scene, spatial_relations=spatial_relations(objs),
                          change_score=round(change, 3), frame_size=(int(frame.shape[1]), int(frame.shape[0])))
        return self.publish_percept(data, sensor)

    def publish_percept(self, data: VisionData, sensor: str) -> Event:
        known = {o.label for o in (self.last_percept or {}).get("objects", [])} if self.last_percept else set()
        labels = {o.label for o in data.objects}
        novelty = len(labels - known) / max(1, len(labels)) if labels else 0.0
        salience = clamp(0.3 + 0.3 * min(1.0, data.change_score * 5) + 0.2 * (data.people > 0) + 0.2 * novelty)
        conf = round(max((o.confidence for o in data.objects), default=0.5), 3)
        text = describe(data.objects, data.scene)
        if self.cfg.keep_frames and self._last_frame is not None:
            self._save_frame()
        self.last_percept = {"t": self.now(), "description": text, "objects": data.objects, "salience": salience}
        return self.emit(EventType.PERCEPTION,
                         PerceptPayload(modality=Modality.VISION, sensor=sensor, description=text, vision=data),
                         summary=text, modality=Modality.VISION, salience=round(salience, 3), confidence=conf,
                         nominated=True, urgency=0.2 if data.people else 0.0)

    async def _vlm(self, frame: np.ndarray) -> str | None:
        if self.ctx.llm.is_symbolic or time.monotonic() - self._last_vlm < self.cfg.vlm_interval_s:
            return None
        import cv2
        self._last_vlm = time.monotonic()
        ok, jpg = cv2.imencode(".jpg", cv2.resize(frame, (512, int(512 * frame.shape[0] / frame.shape[1]))))
        if not ok:
            return None
        res = await self.ctx.llm.complete("describe_scene", "You are a visual perception module.", VLM_PROMPT,
                                          images=[jpg.tobytes()], vision=True, fallback=lambda: "", priority=6,
                                          max_tokens=80)
        return res.text.strip() or None

    def _save_frame(self) -> None:
        import cv2
        out = Path(self.settings.data_dir) / "frames" / f"{new_id('f_')}.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), self._last_frame)

    # ------------------------------------------------------------------ look action
    async def handle(self, event: Event) -> None:
        p = event.data(ActionPayload)
        if p.spec.action != ActionType.LOOK:
            return
        result = {"outcome": "failed", "detail": "no image available: no camera and no image has been shown to me"}
        frame = None
        if self._cap is not None and self._cap.isOpened():
            ok, frame = await asyncio.to_thread(self._cap.read)
            frame = frame if ok else None
        frame = frame if frame is not None else self._last_frame
        if frame is not None:
            ev = await self.process(frame, sensor="camera (deliberate look)", force=True)
            result = {"outcome": "success", "detail": ev.summary if ev else "looked"}
        self.emit(EventType.ACTION_EXECUTED,
                  ActionPayload(decision_id=p.decision_id, spec=p.spec, status="executed", result=result),
                  summary=f"Executed look: {result['detail'][:100]}", modality=Modality.MOTOR)

    def snapshot(self) -> dict:
        lp = self.last_percept
        return {"camera": self.cfg.enabled, "detector": (self.detector.name if self.detector else None),
                "detector_failed": self._detector_failed, "frames": self.frames, "tracking": self.tracking,
                "last": None if not lp else {"t": lp["t"], "description": lp["description"],
                                             "objects": [o.model_dump() for o in lp["objects"]]}}


async def _true(_q: dict) -> bool:
    return True
