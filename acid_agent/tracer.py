"""Append-only tracer: every event goes to a JSONL file AND the Postgres event log (WAL)."""

import json
import time
import uuid
from pathlib import Path

from .config import get_conn


class Tracer:
    def __init__(self, run_id: uuid.UUID, logs_dir: str = "runs"):
        self.run_id = run_id
        self.dir = Path(logs_dir) / str(run_id)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.jsonl = (self.dir / "events.jsonl").open("a", encoding="utf-8")

    def log(self, type: str, **payload):
        rec = {"ts": time.time(), "run_id": str(self.run_id), "type": type, "payload": payload}
        self.jsonl.write(json.dumps(rec, default=str) + chr(10))
        self.jsonl.flush()
        try:
            with get_conn() as conn:
                conn.execute(
                    "INSERT INTO events (run_id, type, payload) VALUES (%s, %s, %s)",
                    (self.run_id, type, json.dumps(payload, default=str)),
                )
        except Exception:
            pass  # tracing must never break execution; JSONL still has it

    def close(self):
        self.jsonl.close()