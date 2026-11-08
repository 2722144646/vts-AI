# Author: 2722144646
# Created: 2026-11-07
# License: AGPL-3.0 (see LICENSE)
"""Diary: one first-person entry per day (or per gap), capped at 1 GB.

Entries are written at the first interaction after a >24h gap, from the day's
episodes and unresolved things. When the folder exceeds the cap the oldest day
files are deleted; a one-line index keeps continuity.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import time

DEFAULT = "default"
DAY_S = 86400.0


class DiaryStore:
    def __init__(self, root: Path, user_id: str = DEFAULT, max_bytes: int = 1024**3) -> None:
        self.root = Path(root) / "diary" / _safe(user_id)
        self.user_id = user_id
        self.max_bytes = max(0, int(max_bytes))

    def _files(self) -> list[Path]:
        if not self.root.exists():
            return []
        return sorted(self.root.glob("*.md"))

    def last_day(self) -> str:
        files = self._files()
        return files[-1].stem if files else ""

    def due(self, now: float | None = None) -> bool:
        """True when the last entry is from an earlier day (or there is none)."""
        last = self.last_day()
        if not last:
            return True
        try:
            last_date = datetime.strptime(last, "%Y-%m-%d").date()
        except ValueError:
            return True
        now_date = datetime.fromtimestamp(time.time() if now is None else now).date()
        return (now_date - last_date).days >= 1

    def write(self, text: str, now: float | None = None) -> Path | None:
        now = time.time() if now is None else now
        day = time.strftime("%Y-%m-%d", time.localtime(now))
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{day}.md"
        if path.exists():
            return path
        path.write_text(text.strip() + "\n", encoding="utf-8")
        self._enforce_cap()
        return path

    def recent(self, count: int = 3) -> list[str]:
        return [path.read_text(encoding="utf-8", errors="replace") for path in self._files()[-count:]]

    def _total_bytes(self) -> int:
        return sum(path.stat().st_size for path in self._files() if path.exists())

    def _enforce_cap(self) -> None:
        if self.max_bytes <= 0:
            return
        files = self._files()
        while files and self._total_bytes() > self.max_bytes:
            oldest = files.pop(0)
            try:
                oldest.unlink()
            except OSError:
                break


def _safe(user_id: str) -> str:
    cleaned = "".join(ch for ch in str(user_id) if ch.isalnum() or ch in "-_")
    return cleaned or DEFAULT
