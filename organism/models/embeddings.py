"""Embedding back-ends.

* ``HashEmbedder``  - deterministic feature-hashing of words and bigrams. Instant; used for
  attention/novelty (the "fast pathway") and as an offline fallback.
* ``OllamaEmbedder`` - neural sentence embeddings (default: nomic-embed-text) for long-term memory.
"""
from __future__ import annotations

import hashlib
import logging

import httpx
import numpy as np

from ..core.util import tokens

log = logging.getLogger("organism.embeddings")


class Embedder:
    name: str = "base"
    dims: int = 0
    relevance_floor: float = 0.2   # similarity below which two texts are considered unrelated

    async def embed(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    async def embed_one(self, text: str) -> np.ndarray:
        return (await self.embed([text]))[0]

    async def close(self) -> None:
        pass


class HashEmbedder(Embedder):
    relevance_floor = 0.12

    def __init__(self, dims: int = 384):
        self.dims = dims
        self.name = f"hash-{dims}"

    def embed_sync(self, text: str) -> np.ndarray:
        v = np.zeros(self.dims, dtype=np.float32)
        toks = tokens(text)
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        for f in feats:
            h = hashlib.blake2b(f.encode(), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "little") % self.dims
            sign = 1.0 if h[4] & 1 else -1.0
            weight = 0.5 if "_" in f else 1.0
            v[idx] += sign * weight
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    async def embed(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.embed_sync(t) for t in texts]) if texts else np.zeros((0, self.dims), np.float32)


class OllamaEmbedder(Embedder):
    relevance_floor = 0.45

    def __init__(self, url: str, model: str, timeout: float = 30.0):
        self.url = url.rstrip("/")
        self.model = model
        self.name = f"ollama:{model}"
        self._client = httpx.AsyncClient(timeout=timeout)
        self.dims = 0

    async def check(self) -> bool:
        try:
            v = await self.embed(["ping"])
            self.dims = int(v.shape[1])
            return True
        except Exception as exc:
            log.warning("ollama embedder unavailable: %r", exc)
            return False

    async def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dims), np.float32)
        r = await self._client.post(f"{self.url}/api/embed", json={"model": self.model, "input": texts})
        r.raise_for_status()
        arr = np.asarray(r.json()["embeddings"], dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.where(norms == 0, 1, norms)

    async def close(self) -> None:
        await self._client.aclose()


class ResilientEmbedder(Embedder):
    """Primary neural embedder that falls back to hashing if the model server disappears.
    The fallback reports its own name so memories are never compared across spaces."""

    def __init__(self, primary: Embedder, fallback: HashEmbedder):
        self.primary = primary
        self.fallback = fallback
        self.name = primary.name
        self.dims = primary.dims
        self.relevance_floor = primary.relevance_floor
        self.degraded = False

    async def embed(self, texts: list[str]) -> np.ndarray:
        if not self.degraded:
            try:
                return await self.primary.embed(texts)
            except Exception as exc:
                log.warning("embedding failed, switching to hashing: %r", exc)
                self.degraded = True
                self.name, self.dims = self.fallback.name, self.fallback.dims
                self.relevance_floor = self.fallback.relevance_floor
        return await self.fallback.embed(texts)

    async def close(self) -> None:
        await self.primary.close()


async def build_embedder(cfg, llm_cfg) -> Embedder:
    fallback = HashEmbedder(cfg.hash_dims)
    if cfg.provider in ("auto", "ollama"):
        oll = OllamaEmbedder(llm_cfg.ollama_url, cfg.ollama_model)
        if await oll.check():
            return ResilientEmbedder(oll, fallback)
        await oll.close()
        if cfg.provider == "ollama":
            log.warning("Ollama embeddings requested but unavailable; using hashing")
    return fallback
