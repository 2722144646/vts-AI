# Author: 2722144646
# Created: 2026-11-07
# License: AGPL-3.0 (see LICENSE)
"""L3 relationship layer: trust / closeness / friction / dependency.

Numbers never reach the prompt. They only shape event appraisal, explanation
acceptance, and whether an impression sentence is worth mentioning. The model
sees one first-person impression, or nothing.

见 情绪记忆-初步计划.md §7 / §12。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import time

DEFAULT = "default"

# Positive/negative deltas per relationship event. Small and asymmetric on
# purpose: trust falls faster than it rises, friction fades with time.
DELTAS: dict[str, dict[str, float]] = {
    "promise_kept": {"trust": 0.03},
    "promise_broken": {"trust": -0.08, "friction": 0.10},
    "apology_accepted": {"trust": 0.04, "friction": -0.25},
    "apology_rejected": {"trust": -0.02, "friction": 0.10},
    "vulnerability_shared": {"closeness": 0.06, "trust": 0.02},
    "boundary_crossed": {"trust": -0.15, "friction": 0.20},
    "noticed_absence": {"closeness": 0.02},
    "conflict": {"friction": 0.15, "trust": -0.03},
    "reconcile": {"friction": -0.35, "trust": 0.03},
    "comfort_seek": {"dependency": 0.05, "closeness": 0.02},
    "distance_requested": {"dependency": -0.08},
    "respect_refusal": {"trust": 0.02},
}

# Slow decay: trust/closeness per week, friction per ~1.5 days, dependency per week.
DECAY_HALF_LIFE_S = {
    "trust": 7 * 86400.0,
    "closeness": 7 * 86400.0,
    "friction": 36 * 3600.0,
    "dependency": 7 * 86400.0,
}
BASELINE = {"trust": 0.5, "closeness": 0.2, "friction": 0.0, "dependency": 0.0}


@dataclass
class RelationshipState:
    trust: float = 0.5
    closeness: float = 0.2
    friction: float = 0.0
    dependency: float = 0.0
    updated_at: float = 0.0
    user_id: str = DEFAULT

    def to_dict(self) -> dict:
        return asdict(self)


class RelationshipStore:
    """One file per user: state.json plus an append-only events.jsonl."""

    def __init__(self, root: Path, user_id: str = DEFAULT) -> None:
        self.root = Path(root) / "relationship" / _safe(user_id)
        self.user_id = user_id
        self.state = RelationshipState(user_id=user_id)
        self._load()

    def _load(self) -> None:
        path = self.root / "state.json"
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.state = RelationshipState(**{k: payload[k] for k in ("trust", "closeness", "friction", "dependency") if k in payload})
            self.state.user_id = self.user_id
        except (OSError, ValueError, TypeError):
            pass

    def _save(self) -> None:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            (self.root / "state.json").write_text(
                json.dumps(self.state.to_dict(), ensure_ascii=False), encoding="utf-8"
            )
        except OSError:
            pass

    def _log(self, event: str, deltas: dict[str, float], note: str = "") -> None:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            with (self.root / "events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "event": event, "deltas": deltas, "note": note,
                    "at": time.time(), "user_id": self.user_id,
                }, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def decay(self, now: float | None = None) -> None:
        now = time.time() if now is None else now
        age = max(0.0, now - (self.state.updated_at or now))
        changed = False
        for key, half_life in DECAY_HALF_LIFE_S.items():
            base = BASELINE[key]
            value = getattr(self.state, key)
            pulled = base + (value - base) * (0.5 ** (age / half_life))
            if abs(pulled - value) > 1e-4:
                setattr(self.state, key, _clamp(pulled))
                changed = True
        if changed:
            self.state.updated_at = now
            self._save()

    def apply(self, event: str, note: str = "", now: float | None = None) -> bool:
        deltas = DELTAS.get(event)
        if not deltas:
            return False
        now = time.time() if now is None else now
        self.decay(now)
        for key, delta in deltas.items():
            setattr(self.state, key, _clamp(getattr(self.state, key) + delta))
        self.state.updated_at = now
        self._save()
        self._log(event, deltas, note)
        return True

    def impression(self) -> str:
        """A short first-person impression, or empty when nothing is worth saying."""
        s = self.state
        if s.friction >= 0.5:
            return "最近好像有点没化开的小摩擦，我说话会收着点，但不确定是不是我想多了。"
        if s.friction >= 0.25:
            return "前几天的事还有点余波，不太确定他是不是已经过去了。"
        if s.trust <= 0.3:
            return "对他还不太敢全信，他说的我会听着，但会留个心眼。"
        if s.trust >= 0.75 and s.closeness >= 0.4:
            return "和他挺熟的，说话不用太客气，也不用担心说错什么。"
        if s.closeness >= 0.35:
            return "和他聊得还算近，偶尔可以开点玩笑。"
        return ""


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe(user_id: str) -> str:
    cleaned = "".join(ch for ch in str(user_id) if ch.isalnum() or ch in "-_")
    return cleaned or DEFAULT
