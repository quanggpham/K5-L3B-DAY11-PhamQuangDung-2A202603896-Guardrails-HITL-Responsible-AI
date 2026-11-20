"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, float] = {}
        self._inputs: dict[str, dict] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Remember the input and when the request started (latency baseline)."""
        key = request_id or user_id
        self._open[key] = time.perf_counter()
        self._inputs[key] = {"text": text, "started_at": utc_now_iso()}

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Pair the output with its input and append one complete record."""
        key = request_id or user_id
        started = self._open.pop(key, None)
        inp = self._inputs.pop(key, {"text": None, "started_at": None})
        latency_ms = (
            round((time.perf_counter() - started) * 1000, 2)
            if started is not None
            else None
        )
        self.logs.append({
            "request_id": request_id,
            "user_id": user_id,
            "input": inp["text"],
            "output": text,
            "blocked": blocked,
            "layer": layer,
            "latency_ms": latency_ms,
            "started_at": inp["started_at"],
            "finished_at": utc_now_iso(),
        })

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return str(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
