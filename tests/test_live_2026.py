"""Live check: does CortexAI look up 2026 facts by itself, and get them right?

These tests use the real internet (Wikipedia), so they are skipped unless asked for:

    CORTEX_LIVE=1 python -m pytest tests/test_live_2026.py -v            # fixed lookup pipeline
    CORTEX_LIVE=1 CORTEX_LIVE_LLM=ollama python -m pytest tests/test_live_2026.py -v   # the agent on a local model

Ground truth was read directly from the Wikipedia articles (September 2026). Nothing is switched on
by hand: the cortex gets the permission from the shipped config, exactly as a normal run would.
"""
import os

import pytest

from organism.config import LLMCfg, load_settings, offline_settings
from organism.organism import Organism
from tests.conftest import run

pytestmark = pytest.mark.skipif(os.environ.get("CORTEX_LIVE") != "1", reason="live internet test (set CORTEX_LIVE=1)")

FACTS_2026 = [
    ("who is the president of usa 2026 ?", ["Trump"]),
    ("who won the champions league title 2026 ?", ["Paris Saint-Germain"]),
    ("Who won the 2026 FIFA World Cup?", ["Spain"]),
    ("Who won Super Bowl LX in 2026?", ["Seahawks"]),
    ("Who won the men's singles at the 2026 Australian Open?", ["Alcaraz"]),
    ("Where were the 2026 Winter Olympics held?", ["Milan", "Cortina", "Ital"]),
    ("Who is the current Secretary-General of the United Nations?", ["Guterres"]),
    ("When was the 2026 Champions League final played?", ["30 May 2026", "May 30, 2026"]),
]


async def live_cortex(tmp_path):
    shipped = load_settings().safety.permissions            # what a normal run would be allowed
    provider = os.environ.get("CORTEX_LIVE_LLM", "rule")
    s = offline_settings(tmp_path, llm=LLMCfg(provider=provider))
    s.safety.permissions.update(shipped)
    org = await Organism(s).start()
    await org.tick(2)
    return org


def test_the_shipped_config_lets_it_look_things_up_by_itself():
    assert load_settings().safety.permissions.get("internet_read") is True


@pytest.mark.parametrize("question,expected", FACTS_2026, ids=[q for q, _ in FACTS_2026])
def test_it_looks_up_2026_facts_by_itself(tmp_path, question, expected):
    async def scenario():
        org = await live_cortex(tmp_path)
        reply = await org.converse(question, max_ticks=40) or ""
        await org.stop()
        assert reply.startswith("I looked it up"), reply
        assert any(e.lower() in reply.lower() for e in expected), reply

    run(scenario())
