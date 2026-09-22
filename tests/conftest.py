import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from organism import Organism, offline_settings  # noqa: E402
from organism.models.llm import Provider  # noqa: E402


def run(coro):
    return asyncio.run(coro)


async def make_organism(tmp_path: Path, disabled: list[str] | None = None, offline_seconds: float = 1.0,
                        **overrides) -> Organism:
    s = offline_settings(tmp_path, **overrides)
    s.modules.disabled = list(disabled or [])
    return await Organism(s, offline_seconds=offline_seconds).start()


class FakeProvider(Provider):
    """Stands in for a neural model: exercises prompt building and JSON parsing without a GPU."""

    name = "fake:model"

    def __init__(self, fail: bool = False, delay: float = 0.0):
        self.fail = fail
        self.delay = delay
        self.prompts: list[str] = []

    async def chat(self, system, prompt, *, schema, images, max_tokens, temperature, vision):
        self.prompts.append(prompt)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise ConnectionError("model server unavailable")
        if schema is None:
            return "Neural reply."
        out = {}
        for key, spec in schema.model_json_schema().get("properties", {}).items():
            if "enum" in spec:
                out[key] = spec["enum"][0]
            elif spec.get("type") == "string":
                out[key] = "a neural thought"
            elif spec.get("type") == "number":
                out[key] = 0.5
            elif spec.get("type") == "boolean":
                out[key] = True
            elif spec.get("type") == "array":
                out[key] = []
        for key, value in (("intention", "answer"), ("kind", "inference"), ("action", "none")):
            if key in out:
                out[key] = value
        return json.dumps(out)


@pytest.fixture
def fake_provider():
    return FakeProvider
