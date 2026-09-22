"""Secret / PII redaction applied before anything is persisted or placed in a model prompt."""
from __future__ import annotations

import re
from typing import Any

_SECRET_PATTERNS = [
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{10,}"),
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9\-]{10,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9\-._~+/]{16,}=*"),
    re.compile(r"(?i)\b(api[_-]?key|secret|password|passwd|token)\b\s*[:=]\s*\S+"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
]
_PII_PATTERNS = [
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    re.compile(r"\+?\d[\d\s().-]{8,}\d"),
]


def redact_text(text: str, pii: bool = False) -> str:
    for pat in _SECRET_PATTERNS:
        text = pat.sub("[REDACTED]", text)
    if pii:
        for pat in _PII_PATTERNS:
            text = pat.sub("[PII]", text)
    return text


def contains_secret(text: str) -> bool:
    return any(p.search(text) for p in _SECRET_PATTERNS)


def redact(obj: Any, pii: bool = False) -> Any:
    if isinstance(obj, str):
        return redact_text(obj, pii)
    if isinstance(obj, dict):
        return {k: redact(v, pii) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(redact(v, pii) for v in obj)
    return obj
