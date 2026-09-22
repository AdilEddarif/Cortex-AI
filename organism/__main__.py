"""Command line interface.

    python -m organism serve                 run continuously + dashboard (http://127.0.0.1:8765)
    python -m organism chat                  talk to the organism in the terminal (real clock)
    python -m organism experiment [names]    run ablation experiments (default: A B C D + baseline)
    python -m organism status                print persisted identity / memory statistics
    python -m organism reset --yes           delete the organism's persistent state
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import sys
from pathlib import Path


def _settings(args):
    from .config import load_settings
    overrides = {}
    if args.data_dir:
        overrides["data_dir"] = Path(args.data_dir)
    s = load_settings(args.config, **overrides)
    if getattr(args, "llm", None):
        s.llm.provider = args.llm
    if getattr(args, "disable", None):
        s.modules.disabled = [m.strip() for m in args.disable.split(",") if m.strip()]
    return s


def cmd_serve(args) -> None:
    import uvicorn
    from .api.server import create_app
    s = _settings(args)
    print(f"CortexAI dashboard: http://{s.api.host}:{args.port or s.api.port}")
    uvicorn.run(create_app(s), host=s.api.host, port=args.port or s.api.port, log_level="warning")


def cmd_chat(args) -> None:
    from .core.events import EventType
    from .organism import Organism

    async def run() -> None:
        s = _settings(args)
        org = await Organism(s).start()
        print(f"[{s.identity.name} | language: {org.ctx.llm.name} | memory: {org.ctx.embedder.name}]")
        print("Type to talk. Commands: /sleep /wake /state /quit. Inner thoughts are shown in grey.\n")

        def show(e) -> None:
            if e.type == EventType.SPEECH_GENERATED:
                print(f"\n\033[96m{s.identity.name}:\033[0m {e.payload.get('text')}\n> ", end="", flush=True)
            elif e.type == EventType.THOUGHT_GENERATED and args.thoughts:
                print(f"\n\033[90m  (thinks) {e.payload.get('content')}\033[0m\n> ", end="", flush=True)
            elif e.type in (EventType.MODE_CHANGED, EventType.DREAM_CONTENT):
                print(f"\n\033[95m  [{e.summary}]\033[0m\n> ", end="", flush=True)

        org.add_observer(show)
        loop = asyncio.get_running_loop()
        try:
            while True:
                line = (await loop.run_in_executor(None, lambda: input("> "))).strip()
                if not line:
                    continue
                if line in ("/quit", "/exit"):
                    break
                if line in ("/sleep", "/wake"):
                    org.command(line[1:])
                elif line == "/state":
                    snap = org.snapshot()["modules"]
                    print(json.dumps({"body": snap.get("brainstem"), "emotion": snap.get("emotion", {}).get("state"),
                                      "workspace": snap.get("workspace", {}).get("items")}, indent=2, default=str))
                else:
                    await org.hear(line)
        except (EOFError, KeyboardInterrupt):
            pass
        finally:
            await org.stop()

    asyncio.run(run())


def cmd_experiment(args) -> None:
    from .experiments.runner import run_experiments
    s = _settings(args)
    names = args.names or ["baseline", "A_no_memory", "B_no_self", "C_no_workspace", "D_deprivation"]
    report = asyncio.run(run_experiments(names, llm=args.neural, seed=args.seed, out_dir=s.data_dir / "experiments"))
    print("\nReports written:", *report.get("files", []), sep="\n  ")


def cmd_status(args) -> None:
    from .persistence.store import Store
    s = _settings(args)
    db = s.data_dir / "organism.db"
    if not db.exists():
        print("No organism state yet at", db)
        return
    st = Store(db)
    lc = st.kv_get("organism.lifecycle", {})
    kinds = st.conn.execute("SELECT kind, COUNT(*) FROM memories WHERE deleted=0 GROUP BY kind").fetchall()
    print(json.dumps({"lifecycle": lc, "memories": dict(kinds), "events": st.count("events"),
                      "transitions": st.count("transitions"), "goals": st.count("goals"),
                      "entities": st.count("entities")}, indent=2, default=str))
    st.close()


def cmd_reset(args) -> None:
    s = _settings(args)
    if not args.yes:
        print(f"This permanently deletes {s.data_dir}. Re-run with --yes to confirm.")
        return
    shutil.rmtree(s.data_dir, ignore_errors=True)
    print("Deleted", s.data_dir)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="organism", description="CortexAI")
    p.add_argument("--config", help="path to a TOML config (default config/organism.toml)")
    p.add_argument("--data-dir", help="persistent state directory")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd")
    sp = sub.add_parser("serve", help="run continuously with the dashboard")
    sp.add_argument("--port", type=int)
    sp.add_argument("--llm", choices=["auto", "ollama", "anthropic", "rule"])
    sp.add_argument("--disable", help="comma-separated modules to ablate")
    sp.set_defaults(fn=cmd_serve)
    sc = sub.add_parser("chat", help="terminal conversation")
    sc.add_argument("--llm", choices=["auto", "ollama", "anthropic", "rule"])
    sc.add_argument("--disable")
    sc.add_argument("--thoughts", action="store_true", help="print inner thoughts")
    sc.set_defaults(fn=cmd_chat)
    se = sub.add_parser("experiment", help="run ablation experiments")
    se.add_argument("names", nargs="*", help="baseline A_no_memory B_no_self C_no_workspace D_deprivation E all ...")
    se.add_argument("--neural", action="store_true", help="use the neural language model (slower)")
    se.add_argument("--seed", type=int, default=7)
    se.set_defaults(fn=cmd_experiment)
    ss = sub.add_parser("status", help="show persisted state")
    ss.set_defaults(fn=cmd_status)
    sr = sub.add_parser("reset", help="delete persistent state")
    sr.add_argument("--yes", action="store_true")
    sr.set_defaults(fn=cmd_reset)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not getattr(args, "fn", None):
        p.print_help()
        sys.exit(0)
    args.fn(args)


if __name__ == "__main__":
    main()
