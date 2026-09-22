"""HTTP/WebSocket interface and developer dashboard.

The organism runs inside the server's event loop (real clock). The dashboard receives a state
snapshot plus the event timeline over a WebSocket; sensory input (text, images, audio) and
operator commands arrive over REST.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from ..config import Settings, load_settings
from ..core.events import Event, EventType
from ..organism import Organism

log = logging.getLogger("organism.api")
STATIC = Path(__file__).parent / "static"
QUIET_TYPES = {EventType.TICK.value, EventType.WM_UPDATED.value, EventType.PREDICTION_MADE.value}


class HearIn(BaseModel):
    text: str
    speaker: str = "user"


class CommandIn(BaseModel):
    command: str
    args: dict = {}


class ExperimentIn(BaseModel):
    conditions: list[str] = ["baseline", "A_no_memory", "B_no_self", "C_no_workspace", "D_deprivation"]
    llm: bool = False
    seed: int = 7


def _event_view(e: Event) -> dict:
    view = {"id": e.id, "t": e.timestamp, "cycle": e.cycle, "type": e.type.value, "source": e.source,
            "summary": e.summary, "modality": e.modality.value, "salience": round(e.salience, 3),
            "confidence": round(e.confidence, 3), "nominated": e.nominated, "caused_by": e.caused_by[-3:]}
    p = e.payload
    if e.type == EventType.SPEECH_GENERATED:   # the face needs the words (and audio, if synthesised)
        view["data"] = {"text": p.get("text"), "intention": p.get("intention"),
                        "audio": Path(p["audio_path"]).name if p.get("audio_path") else None}
    elif e.type == EventType.THOUGHT_GENERATED:
        view["data"] = {"content": p.get("content"), "kind": p.get("kind")}
    elif e.type == EventType.UTTERANCE_UNDERSTOOD:
        view["data"] = {"text": p.get("analysis", {}).get("text"), "speaker": p.get("analysis", {}).get("speaker")}
    return view


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    state: dict = {"organism": None, "clients": set(), "experiment": None}

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        org = Organism(settings)
        await org.start()
        state["organism"] = org

        def fanout(e: Event) -> None:
            if e.type.value in QUIET_TYPES:
                return
            view = _event_view(e)
            for q in list(state["clients"]):
                if q.qsize() < 2000:
                    q.put_nowait(view)

        org.add_observer(fanout)
        try:
            yield
        finally:
            await org.stop()

    app = FastAPI(title="CortexAI", lifespan=lifespan)

    def org() -> Organism:
        o = state["organism"]
        if o is None:
            raise HTTPException(503, "organism not running")
        return o

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/face")
    async def face():
        return FileResponse(STATIC / "face.html")

    @app.get("/brain")
    async def brain():
        return FileResponse(STATIC / "brain.html")

    @app.get("/api/topology")
    async def topology():
        return org().bus.topology()

    @app.get("/api/audio/{name}")
    async def audio(name: str):
        path = (settings.data_dir / "speech" / Path(name).name)
        if not path.exists() or path.suffix != ".wav":
            raise HTTPException(404, "no such audio")
        return FileResponse(path, media_type="audio/wav")

    @app.get("/api/state")
    async def get_state():
        return JSONResponse(json.loads(json.dumps(org().snapshot(), default=str)))

    @app.get("/api/events")
    async def get_events(limit: int = Query(200, le=5000), types: str | None = None):
        return org().store.recent_events(limit, types.split(",") if types else None)

    @app.get("/api/transitions/{event_id}")
    async def get_transitions(event_id: str):
        return org().store.transitions_for(event_id)

    @app.get("/api/memories")
    async def get_memories(q: str | None = None, kind: str | None = None, n: int = 30):
        o = org()
        if "memory" not in o.modules:
            return []
        if q:
            return await o.bus.request_one("memory.recall", {"query": q, "k": n,
                                                              "kinds": [kind] if kind else None,
                                                              "include_imagined": True}, default=[])
        return await o.bus.request_one("memory.recent", {"kinds": [kind] if kind else None, "n": n}, default=[])

    @app.post("/api/hear")
    async def hear(body: HearIn):
        ev = await org().hear(body.text, body.speaker, channel="dashboard")
        return {"event_id": ev.id}

    @app.post("/api/converse")
    async def converse(body: HearIn):
        reply = await org().converse(body.text, body.speaker, timeout=180)
        return {"reply": reply}

    @app.post("/api/see")
    async def see(file: UploadFile = File(...), stream: bool = False):
        try:
            ev = await org().see(await file.read(), sensor="webcam" if stream else "dashboard camera", stream=stream)
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(400, str(exc))
        vis = org().modules.get("vision")
        return {"event_id": ev.id if ev else None, "summary": ev.summary if ev else "no change",
                "tracking": getattr(vis, "tracking", None)}

    @app.post("/api/hear_audio")
    async def hear_audio(file: UploadFile = File(...), speaker: str = "user"):
        try:
            ev = await org().hear_audio(await file.read(), speaker)
        except RuntimeError as exc:
            raise HTTPException(400, str(exc))
        return {"event_id": ev.id if ev else None, "summary": ev.summary if ev else "silence"}

    @app.post("/api/command")
    async def command(body: CommandIn):
        o = org()
        if body.command == "tick":
            o.modules["brainstem"].request_phasic_tick()
        else:
            o.command(body.command, **body.args)
        return {"ok": True}

    @app.post("/api/experiments")
    async def run_experiment(body: ExperimentIn):
        from ..experiments import run_experiments
        if state["experiment"] and not state["experiment"].done():
            raise HTTPException(409, "an experiment is already running")
        out = settings.data_dir / "experiments"
        state["experiment"] = asyncio.create_task(
            run_experiments(body.conditions, llm=body.llm, seed=body.seed, out_dir=out, progress=log.info))
        return {"started": True, "conditions": body.conditions}

    @app.get("/api/experiments")
    async def list_experiments():
        out = settings.data_dir / "experiments"
        task = state["experiment"]
        running = bool(task and not task.done())
        files = sorted(out.glob("experiment-*.json"), reverse=True) if out.exists() else []
        latest = json.loads(files[0].read_text(encoding="utf-8")) if files else None
        err = repr(task.exception()) if task and task.done() and task.exception() else None
        return {"running": running, "error": err, "reports": [f.name for f in files[:20]], "latest": latest}

    @app.websocket("/ws")
    async def ws(socket: WebSocket):
        await socket.accept()
        q: asyncio.Queue = asyncio.Queue()
        state["clients"].add(q)
        try:
            recent = list(reversed(org().store.recent_events(150)))
            await socket.send_text(json.dumps({"kind": "history", "events": [
                {**{k: r[k] for k in ("id", "cycle", "type", "source", "summary", "modality", "salience", "confidence")},
                 "t": r["ts"], "nominated": bool(r["nominated"]), "caused_by": r["caused_by"][-3:]}
                for r in recent if r["type"] not in QUIET_TYPES]}, default=str))
            while True:
                await asyncio.sleep(0.5)
                events = []
                while not q.empty() and len(events) < 300:
                    events.append(q.get_nowait())
                await socket.send_text(json.dumps({"kind": "update", "state": org().snapshot(), "events": events},
                                                  default=str))
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            state["clients"].discard(q)

    return app
