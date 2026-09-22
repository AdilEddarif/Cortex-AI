"""The benchmark measures what it claims, and the demo tells its story."""
import asyncio
import contextlib
import io

from organism.experiments.benchmark import Scenario, _run_one, summarise, task_score


def _run(condition, task, seed=1):
    return asyncio.run(_run_one(condition, seed, task, llm=False))


def test_scenarios_vary_with_the_seed():
    a, b = Scenario.draw(1), Scenario.draw(2)
    assert (a.name, a.like, a.objects) != (b.name, b.like, b.objects)
    assert Scenario.draw(1) == Scenario.draw(1)


def test_each_ablation_breaks_its_own_task():
    for task, ablation in [("attention", "no_attention"), ("prediction", "no_prediction"),
                           ("emotion", "no_emotion"), ("temporal", "no_temporal_memory"),
                           ("memory", "no_memory"), ("action", "no_action_plans")]:
        full, ablated = _run("full", task), _run(ablation, task)
        assert task_score(full["probes"]) == 1.0, (task, full["probes"])
        assert task_score(ablated["probes"]) < 1.0, (task, ablation, ablated["probes"])


def test_summary_pairs_by_seed_and_flags_significant_drops():
    runs = [_run(c, "prediction", s) for c in ("full", "no_prediction") for s in (1, 2, 3)]
    table = summarise(runs, ["prediction"], ["full", "no_prediction"])["table"]
    assert table["full"]["prediction"]["mean"] == 1.0
    cell = table["no_prediction"]["prediction"]
    assert cell["delta"] < 0 and cell["significant"]


def test_demo_tells_the_story():
    from organism.demo import Tour
    tour = Tour(color=False)
    with contextlib.redirect_stdout(io.StringIO()):
        asyncio.run(tour.run())
    said = "\n".join(t for k, t in tour.lines if k == "ai")
    inner = "\n".join(t for k, t in tour.lines if k == "inner")
    assert "the cup is gone" in said and "One, two, three, four, five." in said
    assert "You're Ada" in said and "2nd time" in said and "telescope" in said
    assert "workspace broadcasts: I heard a glass shattering" in inner
    assert sum(k == "scene" for k, _ in tour.lines) == 10
