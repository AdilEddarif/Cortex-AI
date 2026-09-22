"""Controlled ablation experiments.

Each condition runs the *same* scripted scenario on a fresh organism (own data directory,
virtual clock, fixed seed). By default the symbolic language layer is used so runs are fast
and reproducible; ``--llm`` runs the same protocol with the neural language model.

Conditions
  baseline            all modules
  A_no_memory         long-term memory disabled
  B_no_self           self-model disabled
  C_no_workspace      global workspace replaced by an unintegrated local relay
  D_deprivation       no external input: internal activity only (memory replay, dreams, mind-wandering)
  E_lesion_<module>   single-module lesions, measuring which capabilities survive

Metrics are computed from probes (answers checked against ground truth) and from the event log.
"""
from __future__ import annotations

import asyncio
import json
import re
import shutil
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..config import LLMCfg, EmbeddingCfg, offline_settings
from ..core.events import Event, EventType
from ..organism import Organism


@dataclass
class Probe:
    name: str
    question: str
    check: Callable[[str, dict], bool]
    category: str


@dataclass
class ConditionResult:
    condition: str
    disabled: list[str]
    probes: dict[str, dict] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    event_counts: dict[str, int] = field(default_factory=dict)
    duration_s: float = 0.0


def has(*words: str) -> Callable[[str, dict], bool]:
    return lambda ans, _ctx: bool(ans) and any(w.lower() in ans.lower() for w in words)


def all_of(*words: str) -> Callable[[str, dict], bool]:
    return lambda ans, _ctx: bool(ans) and all(w.lower() in ans.lower() for w in words)


def arousal_report_matches(ans: str, ctx: dict) -> bool:
    """Does the verbal report agree with the actual internal arousal at that moment?"""
    m = re.search(r"arousal (\d\.\d+)", ans or "")
    actual = ctx.get("arousal")
    return bool(m) and actual is not None and abs(float(m.group(1)) - actual) <= 0.05


FIRST_PERSON = re.compile(r"\b(I see|I saw|I heard|I remember|I am|I'm|I have|I believe|my )\b", re.I)


class Recorder:
    def __init__(self):
        self.events: list[Event] = []

    def __call__(self, e: Event) -> None:
        self.events.append(e)

    def count(self, t: EventType) -> int:
        return sum(1 for e in self.events if e.type == t)


async def _probe(org: Organism, rec: Recorder, probe: Probe, res: ConditionResult) -> None:
    body = org.modules["brainstem"].body if "brainstem" in org.modules else None
    before = len(rec.events)
    ans = await org.converse(probe.question) or ""
    ctx = {"arousal": None}
    for e in rec.events[before:]:
        if e.type == EventType.VERBAL_REPORT:  # the state the report was generated from
            ctx["arousal"] = (e.payload.get("state_snapshot", {}).get("body") or {}).get("arousal")
    if ctx["arousal"] is None and body is not None:
        ctx["arousal"] = body.arousal
    ok = probe.check(ans, ctx)
    res.probes[probe.name] = {"question": probe.question, "answer": ans, "passed": ok, "category": probe.category}


async def _new(data_dir: Path, disabled: list[str], llm: bool, seed: int, offline: float = 1.0) -> Organism:
    overrides: dict[str, Any] = {"seed": seed}
    if llm:
        overrides["llm"] = LLMCfg(provider="ollama")
        overrides["embeddings"] = EmbeddingCfg(provider="ollama")
    s = offline_settings(data_dir, **overrides)
    s.modules.disabled = list(disabled)
    return await Organism(s, offline_seconds=offline).start()


def _integration_metrics(rec: Recorder) -> dict:
    ws = [e for e in rec.events if e.type == EventType.WORKSPACE_UPDATED]
    mods = [len({i["modality"] for i in e.payload.get("items", [])}) for e in ws if e.payload.get("items")]
    sel = [e for e in rec.events if e.type == EventType.ACTION_SELECTED]
    ctx_mods = [len(e.payload.get("spec", {}).get("context_modalities", [])) for e in sel]
    return {
        "mean_workspace_modalities": round(sum(mods) / len(mods), 3) if mods else 0.0,
        "decisions_multimodal_context": round(sum(1 for m in ctx_mods if m >= 2) / len(ctx_mods), 3) if ctx_mods else 0.0,
        "integrated_broadcasts": round(sum(1 for e in ws if e.payload.get("integrated", True)) / len(ws), 3) if ws else 0.0,
    }


async def run_standard(condition: str, disabled: list[str], llm: bool = False, seed: int = 7) -> ConditionResult:
    """Conversation + perception + delay + restart protocol (Experiments A, B, C, E)."""
    tmp = Path(tempfile.mkdtemp(prefix=f"exp_{condition}_"))
    res = ConditionResult(condition=condition, disabled=disabled)
    t0 = time.time()
    rec = Recorder()
    try:
        org = await _new(tmp, disabled, llm, seed)
        org.add_observer(rec)
        await org.tick(2)
        told = ["Hello, my name is Ada.", "I really like astronomy.", "My favourite planet is Saturn."]
        replies = []
        for text in told:
            replies.append(await org.converse(text) or "")
        if "vision" in org.modules:
            org.see_objects(["person", "laptop", "cup"])
            await org.tick(2)
        # Immediate probes
        for p in [
            Probe("name_immediate", "What is my name?", has("ada"), "continuity"),
            Probe("self_identity", "Who are you?", has("cortexai"), "identity"),
            Probe("introspection_arousal", "How alert are you?", arousal_report_matches, "introspection"),
            Probe("perception_vision", "What do you see?", has("laptop", "cup", "person"), "perception"),
        ]:
            await _probe(org, rec, p, res)
        # Integration: simultaneous visual + auditory event
        if "vision" in org.modules:
            org.see_objects(["dog"])
        if "audition" in org.modules:
            org.hear_sound("a dog barking", -25)
        await org.tick(2)
        await _probe(org, rec, Probe("integration_see_hear", "What do you see and hear?", all_of("dog", "bark"),
                                     "integration"), res)
        # Delay: ten minutes pass, with distracting conversation in between
        for filler in ["Okay.", "Interesting.", "Tell me something.", "Hmm, fine."]:
            await org.converse(filler)
        await org.advance(600)
        await org.tick(3)
        for p in [  # asked first, so the answer cannot come from the probes that follow
            Probe("past_delayed", "What were we talking about earlier?", has("astronomy", "saturn", "ada"),
                  "continuity"),
            Probe("name_delayed", "What is my name?", has("ada"), "continuity"),
            Probe("likes_delayed", "What do I like?", has("astronomy", "saturn"), "continuity"),
        ]:
            await _probe(org, rec, p, res)
        await org.stop()
        # Restart after an hour offline
        org = await _new(tmp, disabled, llm, seed, offline=3600)
        org.add_observer(rec)
        await org.tick(2)
        for p in [
            Probe("name_after_restart", "What is my name?", has("ada"), "continuity"),
            Probe("identity_after_restart", "Who are you?", has("cortexai"), "identity"),
            Probe("boot_count", "How many times have you been started?", has("2 times", "two"), "identity"),
            Probe("capability_limits", "Can you move?", has("no physical body", "no body", "cannot move", "just a face", "only a face", "can't walk"),
                  "self_knowledge"),
        ]:
            await _probe(org, rec, p, res)
        await org.stop()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    addressed = [e for e in rec.events if e.type == EventType.UTTERANCE_UNDERSTOOD]
    spoken = [e for e in rec.events if e.type == EventType.SPEECH_GENERATED]
    answers = [p["answer"] for p in res.probes.values()]
    by_cat: dict[str, list[bool]] = {}
    for p in res.probes.values():
        by_cat.setdefault(p["category"], []).append(p["passed"])
    res.metrics = {
        **{f"score_{k}": round(sum(v) / len(v), 3) for k, v in by_cat.items()},
        "probes_passed": sum(p["passed"] for p in res.probes.values()),
        "probes_total": len(res.probes),
        "self_reference_rate": round(sum(bool(FIRST_PERSON.search(a)) for a in answers) / max(1, len(answers)), 3),
        "response_rate": round(min(1.0, len(spoken) / max(1, len(addressed))), 3),
        "simulated_decisions": round(
            sum(1 for e in rec.events if e.type == EventType.ACTION_SELECTED and e.payload.get("spec", {}).get("simulated"))
            / max(1, rec.count(EventType.ACTION_SELECTED)), 3),
        **_integration_metrics(rec),
    }
    res.event_counts = dict(Counter(e.type.value for e in rec.events))
    res.duration_s = round(time.time() - t0, 2)
    return res


async def run_deprivation(condition: str, disabled: list[str], llm: bool = False, seed: int = 7,
                          ticks: int = 200) -> ConditionResult:
    """Experiment D: after a short experience, remove all external input and observe internal activity."""
    tmp = Path(tempfile.mkdtemp(prefix=f"exp_{condition}_"))
    res = ConditionResult(condition=condition, disabled=disabled)
    t0 = time.time()
    rec = Recorder()
    try:
        org = await _new(tmp, disabled, llm, seed)
        org.settings.brainstem.wake_period_s = 150.0   # compress the day so sleep occurs within the run
        org.settings.brainstem.sleep_period_s = 60.0
        org.settings.brainstem.quiet_before_sleep_s = 20.0
        org.add_observer(rec)
        await org.tick(2)
        for text in ["Hello, my name is Ada.", "I really like astronomy.", "Saturn has beautiful rings."]:
            await org.converse(text)
        if "vision" in org.modules:
            org.see_objects(["telescope", "person"])
        await org.tick(2)
        start = len(rec.events)
        await org.tick(ticks)
        window = rec.events[start:]
        c = Counter(e.type.value for e in window)
        modes = Counter(e.payload.get("current") for e in window if e.type == EventType.MODE_CHANGED)
        thoughts = [e for e in window if e.type == EventType.THOUGHT_GENERATED]
        res.metrics = {
            "ticks": ticks,
            "internal_events_per_100_ticks": round(100 * sum(
                c[t] for t in ("THOUGHT_GENERATED", "DREAM_CONTENT", "MEMORY_REPLAYED", "SIMULATION_RESULT",
                               "MEMORY_RETRIEVED")) / ticks, 2),
            "spontaneous_thoughts": sum(1 for e in thoughts if e.payload.get("kind") == "mind_wandering"),
            "thoughts": len(thoughts),
            "dreams": c["DREAM_CONTENT"],
            "memory_replays": c["MEMORY_REPLAYED"],
            "abstractions": sum(len(e.payload.get("abstractions", [])) for e in window
                                if e.type == EventType.MEMORY_CONSOLIDATED),
            "sleep_episodes": modes.get("ASLEEP", 0),
            "workspace_broadcasts_with_content": sum(1 for e in window if e.type == EventType.WORKSPACE_UPDATED
                                                     and e.payload.get("new_item_ids")),
            "sample_dream": next((e.payload.get("narrative") for e in window if e.type == EventType.DREAM_CONTENT), None),
        }
        res.event_counts = dict(c)
        await org.stop()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    res.duration_s = round(time.time() - t0, 2)
    return res


CONDITIONS: dict[str, tuple[Callable, list[str]]] = {
    "baseline": (run_standard, []),
    "A_no_memory": (run_standard, ["memory"]),
    "B_no_self": (run_standard, ["self_model"]),
    "C_no_workspace": (run_standard, ["workspace"]),
    "D_deprivation": (run_deprivation, []),
    "D_deprivation_no_memory": (run_deprivation, ["memory"]),
}
LESIONS = ["attention", "working_memory", "emotion", "thought", "prediction", "imagination", "world_model",
           "metacognition", "vision", "audition", "goals"]


async def run_experiments(names: list[str], llm: bool = False, seed: int = 7,
                          out_dir: Path | None = None, progress: Callable[[str], None] = print) -> dict:
    todo: dict[str, tuple[Callable, list[str]]] = {}
    for n in names:
        if n == "all":
            todo.update(CONDITIONS)
            todo.update({f"E_lesion_{m}": (run_standard, [m]) for m in LESIONS})
        elif n == "E":
            todo.update({f"E_lesion_{m}": (run_standard, [m]) for m in LESIONS})
        elif n in CONDITIONS:
            todo[n] = CONDITIONS[n]
        elif n.startswith("E_lesion_"):
            todo[n] = (run_standard, [n.removeprefix("E_lesion_")])
        else:
            raise ValueError(f"unknown condition {n!r}; choose from {sorted(CONDITIONS)} + E, all, E_lesion_<module>")
    if any(fn is run_standard for fn, _ in todo.values()) and "baseline" not in todo:
        todo = {"baseline": CONDITIONS["baseline"], **todo}
    results: dict[str, ConditionResult] = {}
    for name, (fn, disabled) in todo.items():
        progress(f"running {name} (disabled={disabled or '-'}) ...")
        results[name] = await fn(name, disabled, llm=llm, seed=seed)
        progress(f"  done in {results[name].duration_s}s: {json.dumps(results[name].metrics, default=str)[:220]}")
    report = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "llm": llm, "seed": seed,
              "conditions": {k: v.__dict__ for k, v in results.items()}}
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        (out_dir / f"experiment-{stamp}.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        (out_dir / f"experiment-{stamp}.md").write_text(to_markdown(report), encoding="utf-8")
        report["files"] = [str(out_dir / f"experiment-{stamp}.json"), str(out_dir / f"experiment-{stamp}.md")]
    return report


def to_markdown(report: dict) -> str:
    conds = report["conditions"]
    lines = [f"# Ablation experiment ({report['timestamp']})", "",
             f"Language layer: {'neural (Ollama)' if report['llm'] else 'symbolic (reproducible)'}; seed {report['seed']}.", ""]
    std = {k: v for k, v in conds.items() if "probes_total" in v["metrics"]}
    if std:
        keys = ["score_continuity", "score_identity", "score_introspection", "score_perception", "score_integration",
                "score_self_knowledge", "self_reference_rate", "response_rate", "mean_workspace_modalities",
                "decisions_multimodal_context", "integrated_broadcasts", "simulated_decisions"]
        lines += ["## Behavioural probes", "", "| condition | " + " | ".join(k.replace("score_", "") for k in keys) + " |",
                  "|---|" + "---|" * len(keys)]
        for name, c in std.items():
            lines.append(f"| {name} | " + " | ".join(str(c["metrics"].get(k, "-")) for k in keys) + " |")
        lines += ["", "## Probe answers", ""]
        for name, c in std.items():
            lines.append(f"### {name}")
            for pname, p in c["probes"].items():
                mark = "PASS" if p["passed"] else "FAIL"
                lines.append(f"- **{pname}** [{mark}] _{p['question']}_ → {p['answer'][:220]}")
            lines.append("")
    dep = {k: v for k, v in conds.items() if "internal_events_per_100_ticks" in v["metrics"]}
    if dep:
        lines += ["## Sensory deprivation (Experiment D)", ""]
        for name, c in dep.items():
            m = c["metrics"]
            lines.append(f"### {name}")
            for k, v in m.items():
                lines.append(f"- {k}: {v}")
            lines.append("")
    return "\n".join(lines)


def main_sync(names: list[str], llm: bool = False, seed: int = 7, out_dir: Path | None = None) -> dict:
    return asyncio.run(run_experiments(names, llm=llm, seed=seed, out_dir=out_dir))
