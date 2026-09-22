from __future__ import annotations

import math
import os
import re
import time
from datetime import datetime

import numpy as np

_STOPWORDS = frozenset(
    "a an the and or but if of to in on at by for with from as is are was were be been being i you he she "
    "it we they me my your our their this that these those do does did have has had not no so what who "
    "where when why how can could would should will just about into than then there here i'm it's don't "
    "that's i'll i've you're let's".split()
)


def new_id(prefix: str = "") -> str:
    """Time-sortable unique id (ms timestamp + random suffix)."""
    return f"{prefix}{int(time.time() * 1000):013x}{os.urandom(4).hex()}"


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


def cosine(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None or a.shape != b.shape:
        return 0.0
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def jaccard_distance(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return 1.0 - len(a & b) / len(a | b)


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def content_words(text: str) -> list[str]:
    return [t for t in tokens(text) if t not in _STOPWORDS and len(t) > 2]


def half_life_decay(dt: float, half_life: float) -> float:
    if half_life <= 0:
        return 0.0
    return 0.5 ** (max(dt, 0.0) / half_life)


def fmt_clock(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")


def fmt_datetime(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def humanize_duration(seconds: float) -> str:
    seconds = abs(seconds)
    if seconds < 60:
        return f"{int(seconds)} seconds"
    if seconds < 3600:
        m = round(seconds / 60)
        return f"{m} minute{'s' if m != 1 else ''}"
    if seconds < 86400:
        h = round(seconds / 3600, 1)
        return "1 hour" if h == 1 else f"{h:g} hours"
    d = round(seconds / 86400, 1)
    return "1 day" if d == 1 else f"{d:g} days"


def truncate(text: str, n: int = 160) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))
