"""Append-only affect log for debugging, replay and blind-test review.

One JSON object per line. Writing must never disturb the performance loop, so
every failure is swallowed after being counted; the bridge keeps running even
if the disk is full or the directory is locked.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import threading
import time

from .schema import AffectEvidence, AffectState


class AffectLog:
    def __init__(self, directory: Path | str, enabled: bool = True) -> None:
        self.directory = Path(directory)
        self.enabled = enabled
        self._lock = threading.Lock()
        self._failures = 0
        self._day = ""
        self._path: Path | None = None

    @property
    def failures(self) -> int:
        return self._failures

    def evidence(self, evidence: AffectEvidence, **extra) -> None:
        payload = evidence.to_dict()
        payload.update(extra)
        self._write("evidence", payload)

    def state(self, state: AffectState, **extra) -> None:
        payload = state.to_dict()
        payload.update(extra)
        self._write("state", payload)

    def event(self, kind: str, payload: dict | None = None, **extra) -> None:
        body = dict(payload or {})
        body.update(extra)
        self._write(kind, body)

    def _write(self, kind: str, payload: dict) -> None:
        if not self.enabled:
            return
        try:
            line = json.dumps(
                {"t": time.time(), "kind": kind, **payload},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            with self._lock:
                path = self._current_path()
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except Exception:
            # Never let bookkeeping break the voice/mouth clock.
            self._failures += 1

    def _current_path(self) -> Path:
        day = datetime.now().strftime("%Y%m%d")
        if self._path is None or day != self._day:
            self.directory.mkdir(parents=True, exist_ok=True)
            self._day = day
            self._path = self.directory / f"affect-{day}.jsonl"
        return self._path


def read_log(path: Path | str) -> list[dict]:
    """Parse a JSONL affect log; malformed lines are skipped, not fatal."""
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows
