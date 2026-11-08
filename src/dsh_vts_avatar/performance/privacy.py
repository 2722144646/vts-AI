# Author: 2722144646
# Created: 2026-11-07
# License: AGPL-3.0 (see LICENSE)
"""AI privacy: the AI defines its own private things and decides when to share.

The store never decides for the model; it only records what was marked private,
how sensitive it is, and whether it has been shared. A trust threshold gates the
default suggestion, and probing below it is a relationship event.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import time
import uuid

DEFAULT = "default"


def trust_required(privacy: float) -> float:
    return max(0.35, min(0.9, 0.35 + 0.55 * max(0.0, min(1.0, privacy))))


@dataclass
class PrivateItem:
    kind: str = "feeling"  # feeling | memory | opinion | fact
    content: str = ""
    privacy: float = 0.5
    shared: bool = False
    created_at: float = 0.0
    shared_at: float = 0.0
    id: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            self.id = f"priv-{uuid.uuid4().hex[:10]}"
        if not self.created_at:
            self.created_at = time.time()

    @property
    def required(self) -> float:
        return trust_required(self.privacy)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["schema_version"] = 1
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "PrivateItem":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in payload.items() if k in known})


class PrivacyStore:
    """JSONL store; the model reads titles, not the whole secret list."""

    def __init__(self, root: Path, user_id: str = DEFAULT) -> None:
        self.path = Path(root) / "private" / f"{_safe(user_id)}.jsonl"
        self.user_id = user_id
        self._items: list[PrivateItem] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                self._items.append(PrivateItem.from_dict(json.loads(line)))
            except (ValueError, TypeError):
                continue

    def _append(self, item: PrivateItem) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
        except OSError:
            pass

    def add(self, kind: str, content: str, privacy: float = 0.5) -> PrivateItem:
        item = PrivateItem(kind=kind, content=content.strip(), privacy=max(0.0, min(1.0, privacy)))
        self._items.append(item)
        self._append(item)
        return item

    def unshared(self) -> list[PrivateItem]:
        return [item for item in self._items if not item.shared]

    def can_share(self, item: PrivateItem, trust: float) -> bool:
        return trust >= item.required

    def mark_shared(self, item_id: str) -> PrivateItem | None:
        item = next((it for it in self._items if it.id == item_id), None)
        if item is None:
            return None
        item.shared = True
        item.shared_at = time.time()
        self._append(item)
        return item


def _safe(user_id: str) -> str:
    cleaned = "".join(ch for ch in str(user_id) if ch.isalnum() or ch in "-_")
    return cleaned or DEFAULT
