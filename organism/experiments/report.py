"""Publishing benchmark results: a Markdown report and light/dark SVG figures."""
from __future__ import annotations

import json
from pathlib import Path

# Reference palette (validated): one sequential blue ramp for scores, categorical slot 1 for bars.
SEQ_LIGHT = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
THEMES = {
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df", "bar": "#2a78d6",
              "na": "#f0efec", "ramp": SEQ_LIGHT},
    "dark": {"surface": "#1a1a19", "text": "#ffffff", "muted": "#c3c2b7", "grid": "#383835", "bar": "#3987e5",
             "na": "#2a2a28", "ramp": SEQ_LIGHT[::-1]},
}
LABELS = {
    "full": "Full CortexAI", "no_memory": "− long-term memory", "no_temporal_memory": "− temporal memory",
    "no_self_model": "− self-model", "no_workspace": "− global workspace", "no_attention": "− attention",
    "no_prediction": "− prediction", "no_emotion": "− emotion", "no_action_plans": "− face / expression",
    "llm_only": "LLM-only chatbot",
}


def _label(c: str) -> str:
    return LABELS.get(c, c)


def _fmt(cell: dict | None) -> str:
    if cell is None:
        return "n/a"
    s = f"{cell['mean']:.2f}"
    if cell.get("significant") and cell.get("delta", 0) < 0:
        s += " ▼"
    return s


def figures(report: dict, out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    out_dir.mkdir(parents=True, exist_ok=True)
    table, tasks = report["summary"]["table"], report["tasks"]
    conds = [c for c in report["conditions"] if c in table]
    written = []
    for theme, th in THEMES.items():
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none",
                             "text.color": th["text"], "axes.labelcolor": th["muted"],
                             "xtick.color": th["muted"], "ytick.color": th["text"]})
        # ---- heatmap: condition x task
        cmap = LinearSegmentedColormap.from_list("seq", th["ramp"])
        fig, ax = plt.subplots(figsize=(1.0 + 0.95 * len(tasks), 0.9 + 0.42 * len(conds)))
        fig.patch.set_facecolor(th["surface"])
        ax.set_facecolor(th["surface"])
        for i, c in enumerate(conds):
            for j, t in enumerate(tasks):
                cell = table[c].get(t)
                if cell is None:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=th["na"], edgecolor=th["surface"], lw=2,
                                               hatch="///", alpha=1.0))
                    ax.text(j + 0.5, i + 0.5, "n/a", ha="center", va="center", color=th["muted"], fontsize=8.5)
                    continue
                v = cell["mean"]
                ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=cmap(v), edgecolor=th["surface"], lw=2))
                strong = v >= 0.55 if theme == "light" else v < 0.55
                ax.text(j + 0.5, i + 0.5, _fmt(cell), ha="center", va="center", fontsize=8.5,
                        color="#ffffff" if strong else "#0b0b0b")
        ax.set_xlim(0, len(tasks))
        ax.set_ylim(len(conds), 0)
        ax.set_xticks([j + 0.5 for j in range(len(tasks))], [t.replace("_", " ") for t in tasks], rotation=30,
                      ha="right")
        ax.set_yticks([i + 0.5 for i in range(len(conds))], [_label(c) for c in conds])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title("Task score by condition (mean over seeds; ▼ = significant drop vs full)", loc="left",
                     fontsize=10.5, color=th["text"], pad=10)
        fig.tight_layout()
        p = out_dir / f"heatmap-{theme}.svg"
        fig.savefig(p, facecolor=th["surface"])
        plt.close(fig)
        written.append(p)
        # ---- overall score with 95% CI
        rows = sorted([c for c in conds if table[c].get("overall")], key=lambda c: table[c]["overall"]["mean"])
        fig, ax = plt.subplots(figsize=(7.2, 0.7 + 0.36 * len(rows)))
        fig.patch.set_facecolor(th["surface"])
        ax.set_facecolor(th["surface"])
        for i, c in enumerate(rows):
            o = table[c]["overall"]
            ax.barh(i, o["mean"], height=0.62, color=th["bar"], edgecolor=th["surface"], linewidth=2)
            ax.plot([o["ci"][0], o["ci"][1]], [i, i], color=th["text"], lw=1.5, solid_capstyle="round")
            ax.text(min(o["ci"][1], 1.0) + 0.015, i, f"{o['mean']:.2f}", va="center", fontsize=9, color=th["text"])
        ax.set_yticks(range(len(rows)), [_label(c) for c in rows])
        ax.set_xlim(0, 1.1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.xaxis.grid(True, color=th["grid"], lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xlabel("overall score (mean of task scores), line = 95% CI")
        ax.set_title("Overall benchmark score", loc="left", fontsize=10.5, color=th["text"], pad=10)
        fig.tight_layout()
        p = out_dir / f"overall-{theme}.svg"
        fig.savefig(p, facecolor=th["surface"])
        plt.close(fig)
        written.append(p)
    return written


def _picture(name: str, alt: str, rel: str = "figures") -> str:
    return (f'<picture>\n  <source media="(prefers-color-scheme: dark)" srcset="{rel}/{name}-dark.svg">\n'
            f'  <img alt="{alt}" src="{rel}/{name}-light.svg">\n</picture>')


def to_markdown(report: dict, fig_rel: str = "figures") -> str:
    s = report["summary"]
    table, tasks, conds = s["table"], report["tasks"], [c for c in report["conditions"] if c in s["table"]]
    rates = s["probe_pass_rates"]
    L = ["# CortexAI benchmark results", "",
         f"Generated {report['timestamp']} by `python -m organism benchmark --publish`. "
         f"{report['seeds']} seeds per cortex condition"
         + (f", {report['llm_seeds']} for the LLM-only baseline" if "llm_only" in conds else "")
         + f"; {len(tasks)} tasks; {report['duration_s']:.0f} s.", "",
         "Every (condition, seed, task) runs on a fresh cortex with a virtual clock. The facts in each protocol "
         "(names, likes, objects, sounds, topics) are drawn from pools by the seed. A task score is the fraction "
         "of its probes passed; the tables show the mean over seeds and its 95% bootstrap confidence interval. "
         "Ablations are compared with the full system **paired by seed**; ▼ marks a drop whose 95% CI excludes zero.",
         "", "## Overall", "", _picture("overall", "Overall benchmark score per condition with 95% CI", fig_rel), "",
         "| condition | overall score | 95% CI |", "|---|---|---|"]
    for c in sorted(conds, key=lambda c: -(table[c].get("overall") or {"mean": -1})["mean"]):
        o = table[c].get("overall")
        if o:
            L.append(f"| {_label(c)} | {o['mean']:.3f} | {o['ci'][0]:.2f} – {o['ci'][1]:.2f} |")
    L += ["", "## By task", "", _picture("heatmap", "Heatmap of task scores by condition", fig_rel), "",
          "| condition | " + " | ".join(t.replace("_", " ") for t in tasks) + " |", "|---|" + "---|" * len(tasks)]
    for c in conds:
        L.append(f"| {_label(c)} | " + " | ".join(_fmt(table[c].get(t)) for t in tasks) + " |")
    L += ["", "## What each ablation breaks", "",
          "Drops that are significant (paired by seed, 95% CI of the difference excludes zero):", ""]
    for c in conds:
        if c == "full":
            continue
        drops = [(t, table[c][t]) for t in tasks if table[c].get(t) and table[c][t].get("significant")
                 and table[c][t].get("delta", 0) < 0]
        if drops:
            L.append(f"- **{_label(c)}**: " + ", ".join(
                f"{t.replace('_', ' ')} {d['delta']:+.2f} [{d['delta_ci'][0]:+.2f}, {d['delta_ci'][1]:+.2f}]"
                for t, d in drops))
        else:
            L.append(f"- **{_label(c)}**: no significant drop on these tasks")
    L += ["", "## Tasks and probes", "", "Pass rate of each probe (full system"
          + (" / LLM-only chatbot" if "llm_only" in conds else "") + ").", ""]
    probe_names: dict[str, list[str]] = {}
    for r in report["runs"]:
        probe_names.setdefault(r["task"], [])
        for p in r["probes"]:
            if p not in probe_names[r["task"]]:
                probe_names[r["task"]].append(p)
    for t in tasks:
        L += [f"### {t.replace('_', ' ')}", "", f"_{report['task_claims'][t]}_", ""]
        for p in probe_names.get(t, []):
            full = rates.get(f"full|{p}")
            llm = rates.get(f"llm_only|{p}") if "llm_only" in conds else None
            cells = f"{full:.0%}" if full is not None else "n/a"
            if "llm_only" in conds:
                cells += " / " + (f"{llm:.0%}" if llm is not None else "n/a")
            L.append(f"- `{p}`: {cells}")
        L.append("")
    L += ["## Example transcripts (seed 1)", ""]
    for c in [x for x in ("full", "llm_only") if x in conds]:
        L += [f"### {_label(c)}", ""]
        for r in report["runs"]:
            if r["condition"] == c and r["seed"] == 1 and r["task"] in ("temporal", "honesty", "self_model"):
                L.append(f"**{r['task'].replace('_', ' ')}**")
                L.append("")
                L += [f"> **Person:** {q}  \n> **{'CortexAI' if c != 'llm_only' else 'Chatbot'}:** {a or '(silent)'}"
                      + "\n>" for q, a in r["transcript"]]
                L.append("")
    L += ["## Caveats", "",
          "- The probes test the architecture's specific claims with scripted protocols. A perfect score means "
          "the claims hold on these protocols, not that the system is generally intelligent.",
          "- Cortex conditions use the deterministic symbolic language layer, so runs are reproducible; the seed "
          "varies the facts and the stochastic parts of cognition.",
          "- The LLM-only baseline is a plain chatbot on a small local model. Probes it structurally cannot attempt "
          "(an internal state to report, a face, sleep) are n/a and excluded from its scores.",
          "- Answers are scored by keyword checks that accept common paraphrases; an unusual but correct phrasing "
          "can still be under-credited. The full transcripts are in the raw JSON for inspection.",
          "- The cortex conditions are deterministic given the facts drawn for a seed, so most outcomes do not vary "
          "across seeds: a zero-width confidence interval means every seed gave the same result.", ""]
    return "\n".join(L)


def publish(report: dict, docs_dir: Path) -> list[Path]:
    docs_dir.mkdir(parents=True, exist_ok=True)
    written = figures(report, docs_dir / "figures")
    md = docs_dir / "results.md"
    md.write_text(to_markdown(report), encoding="utf-8")
    summary = {k: v for k, v in report.items() if k != "runs"}
    js = docs_dir / "results.json"
    js.write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    return written + [md, js]
