"""Live-vision path used by the interactive face: tracking, hysteresis, social orienting."""
import numpy as np
from fastapi.testclient import TestClient

from organism.api.server import create_app
from organism.config import ClockCfg, offline_settings
from organism.core.events import DetectedObject, EventType
from tests.conftest import make_organism, run


class StubDetector:
    name = "stub"

    def __init__(self):
        self.labels = ["person"]

    def detect(self, _frame):
        return [DetectedObject(label=l, confidence=0.9, bbox=(0.4, 0.1, 0.6, 0.9)) for l in self.labels]


def test_stream_hysteresis_and_noticing_a_person(tmp_path):
    async def scenario():
        org = await make_organism(tmp_path)
        vis = org.modules["vision"]
        vis.detector = StubDetector()
        seen = []
        org.add_observer(seen.append)
        await org.tick(2)
        frame = np.zeros((48, 64, 3), np.uint8)
        results = [await vis.process_stream(frame, "webcam") for _ in range(4)]
        assert results[0] is None and results[1] is not None and results[2] is None   # published once, when stable
        assert vis.tracking["objects"][0]["label"] == "person"                         # tracking on every frame
        vis.detector.labels = ["person", "cup"]
        assert await vis.process_stream(frame, "webcam") is None                       # flicker is not a percept
        await org.tick(3)
        speech = [e for e in seen if e.type == EventType.SPEECH_GENERATED]
        assert speech and speech[0].payload["intention"] == "notice_person"
        await org.stop()

    run(scenario())


def test_face_page_and_stream_endpoint(tmp_path):
    s = offline_settings(tmp_path, clock=ClockCfg(mode="real"))
    import cv2
    ok, jpg = cv2.imencode(".jpg", np.zeros((120, 160, 3), np.uint8))
    with TestClient(create_app(s)) as client:
        assert "Wake the face" in client.get("/face").text
        r = client.post("/api/see?stream=true", files={"file": ("f.jpg", jpg.tobytes(), "image/jpeg")})
        assert r.status_code == 200 and "tracking" in r.json()
        assert client.get("/api/audio/../organism.db").status_code == 404
        assert "Brain map" in client.get("/brain").text
        topo = client.get("/api/topology").json()
        assert "expression" in topo["modules"] and "memory" in topo["subscriptions"]["WORKSPACE_UPDATED"]
