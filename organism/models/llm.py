"""Language-model router.

Language models are used as *components* (inner speech, language generation, imagination,
abstraction), never as the container of organism state. Every call site supplies:

  * a ``task`` name (for logging / tracing),
  * a prompt built from structured state,
  * optionally a pydantic ``schema`` for structured output,
  * a deterministic ``fallback`` implementing the same function symbolically.

The fallback is used when the provider is ``rule`` (tests, reproducible experiments) or when
a model call fails/times out, so cognition degrades gracefully instead of stopping.
Prompts are redacted for secrets; credentials never enter a prompt.
"""
from __future__ import annotations

import asyncio
import base64
import heapq
import itertools
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx
from pydantic import BaseModel, ValidationError

from ..core.bus import current_model
from ..core.redact import redact_text

log = logging.getLogger("organism.llm")


@dataclass
class LLMResult:
    text: str
    data: Any = None
    model: str = "rule"
    latency_ms: float = 0.0
    degraded: bool = False


@dataclass
class LLMStats:
    calls: int = 0
    fallbacks: int = 0
    failures: int = 0
    total_ms: float = 0.0
    by_task: dict[str, int] = field(default_factory=dict)
    last_error: str | None = None


class PriorityLock:
    """Single GPU => serialise model calls; lower number = more urgent (speech before dreams)."""

    def __init__(self, capacity: int = 1):
        self._capacity = capacity
        self._busy = 0
        self._heap: list[tuple[int, int, asyncio.Future]] = []
        self._seq = itertools.count()

    async def acquire(self, priority: int) -> None:
        if self._busy < self._capacity and not self._heap:
            self._busy += 1
            return
        fut = asyncio.get_running_loop().create_future()
        heapq.heappush(self._heap, (priority, next(self._seq), fut))
        try:
            await fut
        except asyncio.CancelledError:
            if fut.done() and not fut.cancelled():
                self.release()
            raise

    def release(self) -> None:
        while self._heap:
            _, _, fut = heapq.heappop(self._heap)
            if not fut.done():
                fut.set_result(None)
                return
        self._busy -= 1


def _extract_json(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    if m:
        return m.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start >= 0 and end > start else text


class Provider:
    name = "base"

    async def chat(self, system: str, prompt: str, *, schema: type[BaseModel] | None, images: list[bytes] | None,
                   max_tokens: int, temperature: float, vision: bool) -> str:
        raise NotImplementedError

    async def close(self) -> None:
        pass


class OllamaProvider(Provider):
    def __init__(self, cfg):
        self.cfg = cfg
        self.url = cfg.ollama_url.rstrip("/")
        self.model = cfg.ollama_model
        self.name = f"ollama:{cfg.ollama_model}"
        self._client = httpx.AsyncClient(timeout=cfg.timeout_s)

    async def available(self) -> bool:
        try:
            r = await self._client.get(f"{self.url}/api/tags", timeout=3.0)
            names = {m["name"] for m in r.json().get("models", [])}
            ok = self.cfg.ollama_model in names or f"{self.cfg.ollama_model}:latest" in names
            if not ok:
                log.warning("Ollama running but model %s not pulled", self.cfg.ollama_model)
            return ok
        except Exception:
            return False

    async def chat(self, system, prompt, *, schema, images, max_tokens, temperature, vision) -> str:
        msg: dict[str, Any] = {"role": "user", "content": prompt}
        if images:
            msg["images"] = [base64.b64encode(i).decode() for i in images]
        body: dict[str, Any] = {
            "model": self.cfg.ollama_vision_model if vision else self.model,
            "messages": [{"role": "system", "content": system}, msg],
            "stream": False,
            "think": False,
            "keep_alive": self.cfg.keep_alive,
            "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": self.cfg.num_ctx},
        }
        if schema is not None:
            body["format"] = schema.model_json_schema()
        r = await self._client.post(f"{self.url}/api/chat", json=body)
        r.raise_for_status()
        return r.json()["message"]["content"]

    async def close(self) -> None:
        await self._client.aclose()


class AnthropicProvider(Provider):
    def __init__(self, cfg):
        import anthropic  # optional dependency
        self.cfg = cfg
        self.name = f"anthropic:{cfg.anthropic_model}"
        self._client = anthropic.AsyncAnthropic(timeout=cfg.timeout_s)

    async def chat(self, system, prompt, *, schema, images, max_tokens, temperature, vision) -> str:
        content: list[dict[str, Any]] = []
        for img in images or []:
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/jpeg", "data": base64.b64encode(img).decode()}})
        text = prompt
        if schema is not None:
            text += ("\n\nRespond with only a JSON object matching this JSON schema:\n"
                     + json.dumps(schema.model_json_schema()))
        content.append({"type": "text", "text": text})
        msg = await self._client.messages.create(
            model=self.cfg.anthropic_model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": content}], temperature=temperature,
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")

    async def close(self) -> None:
        await self._client.close()


class LLMRouter:
    def __init__(self, cfg):
        self.cfg = cfg
        self.provider: Provider | None = None
        self.lock = PriorityLock(1)
        self.stats = LLMStats()

    @property
    def name(self) -> str:
        return self.provider.name if self.provider else "rule"

    @property
    def is_symbolic(self) -> bool:
        return self.provider is None

    async def initialize(self) -> str:
        choice = self.cfg.provider
        if choice in ("auto", "ollama"):
            p = OllamaProvider(self.cfg)
            if await p.available():
                self.provider = p
                return self.name
            await p.close()
            if choice == "ollama":
                log.warning("Ollama requested but unavailable; using symbolic fallback")
        if choice in ("auto", "anthropic") and os.environ.get("ANTHROPIC_API_KEY"):
            try:
                self.provider = AnthropicProvider(self.cfg)
                return self.name
            except Exception as exc:
                log.warning("Anthropic provider unavailable: %r", exc)
        self.provider = None
        return self.name

    async def complete(
        self,
        task: str,
        system: str,
        prompt: str,
        *,
        fallback: Callable[[], Any],
        schema: type[BaseModel] | None = None,
        images: list[bytes] | None = None,
        priority: int = 5,
        max_tokens: int = 300,
        temperature: float | None = None,
        vision: bool = False,
    ) -> LLMResult:
        self.stats.calls += 1
        self.stats.by_task[task] = self.stats.by_task.get(task, 0) + 1
        models = current_model.get()

        def use_fallback(degraded: bool) -> LLMResult:
            value = fallback()
            if models is not None:
                models.append("rule")
            if degraded:
                self.stats.fallbacks += 1
            if isinstance(value, BaseModel):
                return LLMResult(text=value.model_dump_json(), data=value, model="rule", degraded=degraded)
            return LLMResult(text=str(value), data=value, model="rule", degraded=degraded)

        if self.provider is None:
            return use_fallback(False)

        system, prompt = redact_text(system), redact_text(prompt)
        t0 = time.perf_counter()
        await self.lock.acquire(priority)
        try:
            text = await asyncio.wait_for(
                self.provider.chat(system, prompt, schema=schema, images=images, max_tokens=max_tokens,
                                   temperature=self.cfg.temperature if temperature is None else temperature,
                                   vision=vision),
                timeout=self.cfg.timeout_s,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.stats.failures += 1
            self.stats.last_error = f"{task}: {exc!r}"[:300]
            log.warning("LLM task %s failed (%r); using symbolic fallback", task, exc)
            return use_fallback(True)
        finally:
            self.lock.release()
        dt = (time.perf_counter() - t0) * 1000.0
        self.stats.total_ms += dt
        if models is not None:
            models.append(self.provider.name)

        if schema is not None:
            try:
                data = schema.model_validate_json(_extract_json(text))
            except (ValidationError, ValueError) as exc:
                log.warning("LLM task %s returned invalid JSON (%r)", task, exc)
                return use_fallback(True)
            return LLMResult(text=text, data=data, model=self.provider.name, latency_ms=dt)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
        if not text:
            return use_fallback(True)
        return LLMResult(text=text, data=text, model=self.provider.name, latency_ms=dt)

    def snapshot(self) -> dict[str, Any]:
        s = self.stats
        return {
            "provider": self.name, "calls": s.calls, "fallbacks": s.fallbacks, "failures": s.failures,
            "avg_ms": round(s.total_ms / max(1, s.calls - s.fallbacks), 1) if self.provider else 0.0,
            "by_task": dict(s.by_task), "last_error": s.last_error,
        }

    async def close(self) -> None:
        if self.provider:
            await self.provider.close()
