"""Experiment framework and HTTP interface."""
import time

from fastapi.testclient import TestClient

from organism.api.server import create_app
from organism.config import offline_settings, ClockCfg
from organism.experiments.runner import run_deprivation, run_standard
from tests.conftest import run


def test_memory_ablation_reduces_continuity():
    base = run(run_standard("baseline", []))
    amnesic = run(run_standard("A_no_memory", ["memory"]))
    assert base.metrics["score_continuity"] == 1.0
    assert amnesic.metrics["score_continuity"] < base.metrics["score_continuity"]
    assert not amnesic.probes["name_after_restart"]["passed"]


def test_workspace_ablation_reduces_integration():
    base = run(run_standard("baseline", []))
    local = run(run_standard("C_no_workspace", ["workspace"]))
    assert local.metrics["mean_workspace_modalities"] < base.metrics["mean_workspace_modalities"]
    assert local.metrics["integrated_broadcasts"] == 0.0


def test_deprivation_produces_internal_activity():
    res = run(run_deprivation("D", [], ticks=150))
    assert res.metrics["dreams"] > 0
    assert res.metrics["internal_events_per_100_ticks"] > 0


def test_api_roundtrip(tmp_path):
    s = offline_settings(tmp_path, clock=ClockCfg(mode="real"))
    s.brainstem.tick_max_s = 0.3
    s.brainstem.tick_min_s = 0.1
    with TestClient(create_app(s)) as client:
        assert client.get("/").status_code == 200
        r = client.post("/api/converse", json={"text": "Hello, my name is Ada."})
        assert r.status_code == 200 and "Ada" in (r.json()["reply"] or "")
        state = client.get("/api/state").json()
        assert state["modules"]["brainstem"]["mode"] == "AWAKE"
        assert client.get("/api/memories", params={"q": "name"}).status_code == 200
        with client.websocket_connect("/ws") as ws:
            first = ws.receive_json()
            assert first["kind"] == "history"
            deadline = time.time() + 5
            upd = ws.receive_json()
            assert upd["kind"] == "update" and time.time() < deadline
