"""Web sources for the research agent: Wikipedia and Wikidata (free, open) and Tavily (web search, API key).

Only impersonal questions ever reach these tools (the knowledge module filters everything about
the organism, the speaker or the present moment before any call). Every call has a timeout, a
cache and an hourly budget; failures return ``None`` so cognition carries on ("I don't know").
The Tavily key is read from the environment (``TAVILY_API_KEY``) and never enters a prompt or log.
"""
from __future__ import annotations

import logging
import os
import time
from collections import deque
from typing import Any

import httpx

log = logging.getLogger("organism.web")

USER_AGENT = "CortexAI/0.1 (research agent; https://github.com/AdilEddarif/Cortex-AI)"


class WebTools:
    def __init__(self, lang: str = "en", timeout_s: float = 10.0, cache_ttl_s: float = 86400.0,
                 max_calls_per_hour: int = 30, transport: httpx.AsyncBaseTransport | None = None,
                 tavily_key: str | None = None):
        self.lang = lang
        self.cache_ttl_s = cache_ttl_s
        self.max_calls_per_hour = max_calls_per_hour
        self._key = tavily_key if tavily_key is not None else os.environ.get("TAVILY_API_KEY", "")
        self._client = httpx.AsyncClient(timeout=timeout_s, transport=transport,
                                         headers={"User-Agent": USER_AGENT})
        self._cache: dict[str, tuple[float, Any]] = {}
        self._calls: deque[float] = deque()
        self.stats = {"calls": 0, "cache_hits": 0, "failures": 0, "budget_refusals": 0}

    @property
    def has_web_search(self) -> bool:
        return bool(self._key)

    async def close(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------ plumbing
    def _budget_ok(self) -> bool:
        now = time.monotonic()
        while self._calls and now - self._calls[0] > 3600:
            self._calls.popleft()
        if len(self._calls) >= self.max_calls_per_hour:
            self.stats["budget_refusals"] += 1
            return False
        self._calls.append(now)
        return True

    async def _get(self, key: str, fn) -> Any:
        hit = self._cache.get(key)
        if hit and time.monotonic() - hit[0] < self.cache_ttl_s:
            self.stats["cache_hits"] += 1
            return hit[1]
        if not self._budget_ok():
            return None
        self.stats["calls"] += 1
        try:
            value = await fn()
        except Exception as exc:  # network trouble is not a reason to stop thinking
            self.stats["failures"] += 1
            log.warning("web call %s failed: %r", key.split(":", 1)[0], exc)
            return None
        self._cache[key] = (time.monotonic(), value)
        return value

    # ------------------------------------------------------------------ Wikipedia
    async def wikipedia_search(self, query: str, n: int = 3) -> list[dict] | None:
        async def call():
            r = await self._client.get(f"https://{self.lang}.wikipedia.org/w/api.php", params={
                "action": "query", "list": "search", "srsearch": query, "srlimit": n, "format": "json"})
            r.raise_for_status()
            return [{"title": h["title"], "snippet": _strip_html(h.get("snippet", ""))}
                    for h in r.json().get("query", {}).get("search", [])]
        return await self._get(f"wsearch:{query.lower()}", call)

    async def wikipedia_page(self, title: str) -> dict | None:
        """The whole introduction of an article (plain text), its URL and when it was last edited."""
        async def call():
            r = await self._client.get(f"https://{self.lang}.wikipedia.org/w/api.php", params={
                "action": "query", "prop": "extracts|info", "exintro": 1, "explaintext": 1, "redirects": 1,
                "inprop": "url", "titles": title, "format": "json", "formatversion": 2})
            r.raise_for_status()
            pages = r.json().get("query", {}).get("pages", [])
            d = pages[0] if pages else {}
            if not d or d.get("missing"):
                return {}
            return {"title": d.get("title", title), "text": d.get("extract", "").strip(),
                    "url": d.get("fullurl", ""), "date": (d.get("touched") or "")[:10], "source": "Wikipedia"}
        return await self._get(f"wpage:{title.lower()}", call)

    # ------------------------------------------------------------------ Wikidata
    async def wikidata_officeholder(self, office: str, year: int | None = None) -> dict | None:
        """Who holds (or held, in ``year``) an office such as "President of France", from Wikidata's
        officeholder statements (P1308) with their start (P580) and end (P582) dates."""
        async def call():
            api = "https://www.wikidata.org/w/api.php"
            r = await self._client.get(api, params={"action": "wbsearchentities", "search": office, "language": "en",
                                                    "type": "item", "limit": 3, "format": "json"})
            r.raise_for_status()
            for cand in r.json().get("search", []):
                qid = cand["id"]
                r = await self._client.get(api, params={"action": "wbgetentities", "ids": qid, "props": "claims|labels",
                                                        "languages": "en|mul", "format": "json"})
                r.raise_for_status()
                ent = r.json().get("entities", {}).get(qid, {})
                holders = []
                for c in ent.get("claims", {}).get("P1308", []):
                    if c.get("rank") == "deprecated":
                        continue
                    value = ((c.get("mainsnak") or {}).get("datavalue") or {}).get("value") or {}
                    q = c.get("qualifiers", {})
                    start = _wd_time(q.get("P580"))
                    end = _wd_time(q.get("P582"))
                    if value.get("id"):
                        holders.append({"id": value["id"], "start": start, "end": end})
                if not holders:
                    continue
                if year is None:  # in office today (an end date may be a scheduled end of term)
                    today = time.strftime("%Y-%m-%d")
                    pool = [h for h in holders if (h["start"] or "0000") <= today and (not h["end"] or h["end"] >= today)]
                else:
                    pool = [h for h in holders if (h["start"] or "0000")[:4] <= str(year)
                            and (not h["end"] or h["end"][:4] >= str(year))]
                if not pool:
                    return {}
                best = max(pool, key=lambda h: h["start"] or "")
                r = await self._client.get(api, params={"action": "wbgetentities", "ids": best["id"], "props": "labels",
                                                        "languages": "en|mul", "format": "json"})
                r.raise_for_status()
                ents = r.json().get("entities", {})
                person = ents.get(best["id"]) or next(iter(ents.values()), {})  # the id may redirect
                return {"office": _label(ent) or office, "holder": _label(person),
                        "start": best["start"], "end": best["end"],
                        "url": f"https://www.wikidata.org/wiki/{qid}", "source": "Wikidata",
                        "date": time.strftime("%Y-%m-%d")}
            return {}
        return await self._get(f"wdoffice:{office.lower()}:{year}", call)

    # ------------------------------------------------------------------ Tavily
    async def web_search(self, query: str, n: int = 5) -> dict | None:
        if not self._key:
            return None

        async def call():
            r = await self._client.post("https://api.tavily.com/search", json={
                "query": query, "search_depth": "basic", "include_answer": True, "max_results": n},
                headers={"Authorization": f"Bearer {self._key}"})
            r.raise_for_status()
            d = r.json()
            return {"answer": d.get("answer") or "",
                    "results": [{"title": x.get("title", ""), "url": x.get("url", ""),
                                 "text": (x.get("content") or "")[:600], "date": (x.get("published_date") or "")[:10],
                                 "source": _domain(x.get("url", ""))} for x in d.get("results", [])]}
        return await self._get(f"tavily:{query.lower()}", call)


def _label(entity: dict) -> str:
    """An entity's English name; names shared by all languages are stored as "mul" (multilingual)."""
    labels = entity.get("labels", {})
    return (labels.get("en") or labels.get("mul") or {}).get("value", "")


def _wd_time(snaks: list | None) -> str | None:
    """A Wikidata time qualifier ("+2025-01-20T00:00:00Z") as "2025-01-20"."""
    for s in snaks or []:
        v = ((s.get("datavalue") or {}).get("value") or {}).get("time")
        if v:
            return v.lstrip("+")[:10]
    return None


def _strip_html(s: str) -> str:
    import re
    return re.sub(r"<[^>]+>", "", s).replace("&quot;", '"').replace("&amp;", "&")


def _domain(url: str) -> str:
    host = url.split("//", 1)[-1].split("/", 1)[0]
    return host.removeprefix("www.")
