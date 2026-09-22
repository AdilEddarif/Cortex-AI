"""Structured persistent storage (SQLite, WAL).

Tables:
  events       - the full cognitive event log (batched writes)
  transitions  - reproducibility records (module, input, outputs, state before/after, model, latency)
  memories     - long-term memory records with embeddings
  goals        - goal records
  entities     - world-model entities
  kv           - small persistent state (identity, self-model, brainstem, clock...)
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..core.events import Event
from ..core.redact import redact

SCHEMA = """
CREATE TABLE IF NOT EXISTS events(
  id TEXT PRIMARY KEY, ts REAL, cycle INTEGER, type TEXT, source TEXT, modality TEXT,
  summary TEXT, confidence REAL, salience REAL, importance REAL, urgency REAL,
  emotional_value REAL, uncertainty REAL, nominated INTEGER, caused_by TEXT, payload TEXT,
  run_label TEXT);
CREATE INDEX IF NOT EXISTS events_ts ON events(ts);
CREATE INDEX IF NOT EXISTS events_type ON events(type);
CREATE TABLE IF NOT EXISTS transitions(
  id INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT, ts REAL, module TEXT, input_type TEXT,
  input_summary TEXT, outputs TEXT, state_before TEXT, state_after TEXT, state_digest TEXT,
  model_used TEXT, latency_ms REAL, confidence REAL, run_label TEXT);
CREATE INDEX IF NOT EXISTS transitions_event ON transitions(event_id);
CREATE TABLE IF NOT EXISTS memories(
  id TEXT PRIMARY KEY, kind TEXT, content TEXT, ts REAL, importance REAL, confidence REAL,
  source TEXT, context TEXT, emotion TEXT, episode_id TEXT, access_count INTEGER DEFAULT 0,
  last_access REAL, consolidated INTEGER DEFAULT 0, strength REAL DEFAULT 1.0,
  embedding BLOB, embed_model TEXT, tags TEXT, deleted INTEGER DEFAULT 0);
CREATE INDEX IF NOT EXISTS memories_kind ON memories(kind);
CREATE TABLE IF NOT EXISTS goals(id TEXT PRIMARY KEY, status TEXT, data TEXT, updated REAL);
CREATE TABLE IF NOT EXISTS entities(id TEXT PRIMARY KEY, data TEXT, updated REAL);
CREATE TABLE IF NOT EXISTS kv(key TEXT PRIMARY KEY, value TEXT, updated REAL);
"""

MEMORY_COLUMNS = (
    "id", "kind", "content", "ts", "importance", "confidence", "source", "context", "emotion",
    "episode_id", "access_count", "last_access", "consolidated", "strength", "embedding",
    "embed_model", "tags",
)


class Store:
    def __init__(self, path: str | Path, run_label: str = "default", redact_pii: bool = False,
                 flush_every: int = 200):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_label = run_label
        self.redact_pii = redact_pii
        self.flush_every = flush_every
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(SCHEMA)
        self._event_buf: list[tuple] = []
        self._trans_buf: list[tuple] = []

    # ------------------------------------------------------------------ event log
    def log_event(self, e: Event) -> None:
        payload = redact(e.payload, self.redact_pii)
        self._event_buf.append((
            e.id, e.timestamp, e.cycle, e.type.value, e.source, e.modality.value,
            redact(e.summary, self.redact_pii), e.confidence, e.salience, e.importance, e.urgency,
            e.emotional_value, e.uncertainty, int(e.nominated), json.dumps(e.caused_by),
            json.dumps(payload, default=str), self.run_label,
        ))
        if len(self._event_buf) >= self.flush_every:
            self.flush()

    def log_transition(self, rec: dict[str, Any]) -> None:
        self._trans_buf.append((
            rec["event_id"], rec["ts"], rec["module"], rec["input_type"],
            redact(rec["input_summary"], self.redact_pii), rec["outputs"],
            redact(rec["state_before"], self.redact_pii) if rec["state_before"] else None,
            redact(rec["state_after"], self.redact_pii) if rec["state_after"] else None,
            rec["state_digest"], rec["model_used"], rec["latency_ms"], rec["confidence"],
            self.run_label,
        ))
        if len(self._trans_buf) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        with self._lock:
            if not self._event_buf and not self._trans_buf:
                return
            ev, tr = self._event_buf, self._trans_buf
            self._event_buf, self._trans_buf = [], []
            self.conn.execute("BEGIN")
            try:
                if ev:
                    self.conn.executemany(
                        "INSERT OR REPLACE INTO events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", ev)
                if tr:
                    self.conn.executemany(
                        "INSERT INTO transitions(event_id, ts, module, input_type, input_summary, outputs,"
                        " state_before, state_after, state_digest, model_used, latency_ms, confidence,"
                        " run_label) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", tr)
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise

    def recent_events(self, limit: int = 200, types: Iterable[str] | None = None,
                      since: float | None = None) -> list[dict]:
        self.flush()
        q = "SELECT id, ts, cycle, type, source, modality, summary, confidence, salience, importance," \
            " urgency, emotional_value, nominated, caused_by, payload FROM events"
        cond, args = [], []
        if types:
            types = list(types)
            cond.append(f"type IN ({','.join('?' * len(types))})")
            args += types
        if since is not None:
            cond.append("ts >= ?")
            args.append(since)
        if cond:
            q += " WHERE " + " AND ".join(cond)
        q += " ORDER BY ts DESC, rowid DESC LIMIT ?"
        args.append(limit)
        cols = ["id", "ts", "cycle", "type", "source", "modality", "summary", "confidence", "salience",
                "importance", "urgency", "emotional_value", "nominated", "caused_by", "payload"]
        with self._lock:
            rows = self.conn.execute(q, args).fetchall()
        out = []
        for r in rows:
            d = dict(zip(cols, r))
            d["caused_by"] = json.loads(d["caused_by"] or "[]")
            d["payload"] = json.loads(d["payload"] or "{}")
            out.append(d)
        return out

    def transitions_for(self, event_id: str) -> list[dict]:
        self.flush()
        with self._lock:
            cur = self.conn.execute("SELECT * FROM transitions WHERE event_id=?", (event_id,))
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]

    def count(self, table: str) -> int:
        self.flush()
        with self._lock:
            return self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    # ------------------------------------------------------------------ memories
    def put_memory(self, rec: dict[str, Any]) -> None:
        row = dict(rec)
        emb = row.get("embedding")
        row["embedding"] = emb.astype(np.float32).tobytes() if isinstance(emb, np.ndarray) else None
        for k in ("context", "emotion", "tags"):
            row[k] = json.dumps(redact(row.get(k) or ({} if k != "tags" else []), self.redact_pii))
        row["content"] = redact(row["content"], self.redact_pii)
        vals = [row.get(c) for c in MEMORY_COLUMNS]
        with self._lock:
            self.conn.execute(
                f"INSERT OR REPLACE INTO memories({','.join(MEMORY_COLUMNS)}) VALUES"
                f" ({','.join('?' * len(MEMORY_COLUMNS))})", vals)

    def update_memory(self, mem_id: str, **fields: Any) -> None:
        if not fields:
            return
        sets, args = [], []
        for k, v in fields.items():
            if k in ("context", "emotion", "tags"):
                v = json.dumps(v)
            elif k == "embedding" and isinstance(v, np.ndarray):
                v = v.astype(np.float32).tobytes()
            sets.append(f"{k}=?")
            args.append(v)
        args.append(mem_id)
        with self._lock:
            self.conn.execute(f"UPDATE memories SET {', '.join(sets)} WHERE id=?", args)

    def load_memories(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                f"SELECT {','.join(MEMORY_COLUMNS)} FROM memories WHERE deleted=0").fetchall()
        out = []
        for r in rows:
            d = dict(zip(MEMORY_COLUMNS, r))
            d["embedding"] = np.frombuffer(d["embedding"], dtype=np.float32).copy() if d["embedding"] else None
            for k in ("context", "emotion", "tags"):
                d[k] = json.loads(d[k] or ("[]" if k == "tags" else "{}"))
            out.append(d)
        return out

    def delete_memory(self, mem_id: str) -> None:
        with self._lock:
            self.conn.execute("UPDATE memories SET deleted=1 WHERE id=?", (mem_id,))

    # ------------------------------------------------------------------ goals / entities
    def put_goal(self, goal_id: str, status: str, data: dict, ts: float) -> None:
        with self._lock:
            self.conn.execute("INSERT OR REPLACE INTO goals VALUES (?,?,?,?)",
                              (goal_id, status, json.dumps(data), ts))

    def load_goals(self) -> list[dict]:
        with self._lock:
            return [json.loads(r[0]) for r in self.conn.execute("SELECT data FROM goals").fetchall()]

    def put_entity(self, entity_id: str, data: dict, ts: float) -> None:
        with self._lock:
            self.conn.execute("INSERT OR REPLACE INTO entities VALUES (?,?,?)",
                              (entity_id, json.dumps(redact(data, self.redact_pii), default=str), ts))

    def load_entities(self) -> list[dict]:
        with self._lock:
            return [json.loads(r[0]) for r in self.conn.execute("SELECT data FROM entities").fetchall()]

    def delete_entity(self, entity_id: str) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM entities WHERE id=?", (entity_id,))

    # ------------------------------------------------------------------ kv
    def kv_get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            row = self.conn.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def kv_set(self, key: str, value: Any, ts: float = 0.0) -> None:
        with self._lock:
            self.conn.execute("INSERT OR REPLACE INTO kv VALUES (?,?,?)",
                              (key, json.dumps(value, default=str), ts))

    def close(self) -> None:
        self.flush()
        with self._lock:
            self.conn.close()
