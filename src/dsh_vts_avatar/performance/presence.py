"""Presence director: what she does when nobody is talking.

Idle behaviour is not random animation — it is chosen from how she feels
(mood), how close the relationship is, and what the diary says. The action
batches themselves are written by the model: when a threshold is crossed (or
after 24 hours without a new batch), the director asks for a fresh batch.

Library file: ``data/presence/idle_library.json``::

    {
      "generated_at": 1760000000.0,
      "note": "…",
      "profiles": {
        "bright": {"actions": [{"kind": "idle_tilt", "weight": 2.0, "intensity": 0.6}]}
      }
    }
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# Kinds whose animation touches mouth parameters. They are allowed at idle but
# must yield immediately when speech starts.
MOUTH_KINDS = {"idle_sigh", "idle_hum", "idle_yawn"}

# Default pools per profile; also defines the set of legal idle kinds.
POOLS: dict[str, dict[str, float]] = {
    "calm": {"idle_glance": 3.0, "idle_tilt": 2.0, "idle_nod": 1.0, "idle_breathe": 2.0},
    "cozy": {"idle_look_user": 3.0, "idle_tilt": 2.0, "idle_sigh": 1.0, "idle_hum": 1.0},
    "bright": {"idle_glance": 2.0, "idle_tilt": 2.5, "idle_nod": 1.5, "idle_hum": 1.5},
    "low": {"idle_look_down": 3.0, "idle_sigh": 2.0, "idle_tilt": 1.0, "idle_breathe": 2.0},
    "tense": {"idle_look_away": 3.0, "idle_breathe": 2.0, "idle_tilt": 1.0, "idle_nod": 0.5},
}

KNOWN_IDLE_KINDS = set().union(*(set(pool) for pool in POOLS.values())) | MOUTH_KINDS | {
    "idle_look_down",
    "idle_look_away",
    "idle_look_user",
}

MIN_GAP = 6.0
MAX_GAP = 12.0
BATCH_TTL_S = 24 * 3600.0


@dataclass
class PresenceProfile:
    key: str = "calm"
    energy: float = 0.55
    warmth: float = 0.3
    tension: float = 0.0
    mouth: float = 0.2
    note: str = ""


@dataclass
class PresenceDirector:
    """Chooses idle cues from mood + relationship + diary, using model batches."""

    base_energy: float = 0.55
    rng: random.Random = field(default_factory=random.Random)
    library_path: str = ""
    profile: PresenceProfile = field(default_factory=PresenceProfile)
    library: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    generated_at: float = 0.0
    needs_write: bool = True
    write_reason: str = "never"
    library_note: str = ""
    _next_at: float = 0.0
    _last: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._load()

    # -- library ----------------------------------------------------------

    def _load(self) -> None:
        if not self.library_path:
            return
        try:
            raw = json.loads(Path(self.library_path).read_text(encoding="utf-8"))
        except Exception:
            return
        self.generated_at = float(raw.get("generated_at") or 0.0)
        self.library_note = str(raw.get("note") or "")
        library: dict[str, dict[str, dict[str, float]]] = {}
        for key, entry in (raw.get("profiles") or {}).items():
            actions = {}
            for item in (entry or {}).get("actions") or []:
                kind = str(item.get("kind") or "")
                if kind not in KNOWN_IDLE_KINDS:
                    continue
                weight = float(item.get("weight", 1.0))
                if weight <= 0:
                    continue
                actions[kind] = {
                    "w": min(5.0, max(0.1, weight)),
                    "i": min(1.2, max(0.1, float(item.get("intensity", 0.6)))),
                }
            if actions:
                library[str(key)] = actions
        self.library = library
        if self.library:
            self.needs_write = False
            self.write_reason = ""

    def save_batch(self, profile: str, actions: list[dict[str, Any]], note: str = "") -> dict[str, Any]:
        """Store a model-written batch for one profile and clear the refresh flag."""
        profile = str(profile or "").strip().lower()
        if profile not in POOLS:
            raise ValueError(f"unknown profile: {profile}")
        cleaned: dict[str, dict[str, float]] = {}
        for item in actions or []:
            kind = str(item.get("kind") or "")
            if kind not in KNOWN_IDLE_KINDS:
                raise ValueError(f"unknown idle action: {kind}")
            weight = float(item.get("weight", 1.0))
            if weight <= 0:
                continue
            cleaned[kind] = {
                "w": min(5.0, max(0.1, weight)),
                "i": min(1.2, max(0.1, float(item.get("intensity", 0.6)))),
            }
        if not cleaned:
            raise ValueError("empty batch")
        self.library[profile] = cleaned
        self.generated_at = time.time()
        self.needs_write = False
        self.write_reason = ""
        self.library_note = str(note or "")[:200]
        self._save()
        return {
            "ok": True,
            "profile": profile,
            "actions": len(cleaned),
            "generated_at": self.generated_at,
        }

    def _save(self) -> None:
        if not self.library_path:
            return
        path = Path(self.library_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": self.generated_at,
            "note": self.library_note,
            "profiles": {
                key: {
                    "actions": [
                        {"kind": kind, "weight": value["w"], "intensity": value["i"]}
                        for kind, value in pool.items()
                    ]
                }
                for key, pool in self.library.items()
            },
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def check_expiry(self, now: float | None = None) -> bool:
        """Ask for a fresh batch after 24 hours without one."""
        now = time.time() if now is None else now
        if self.generated_at <= 0:
            self.needs_write = True
            self.write_reason = self.write_reason or "never"
            return True
        if now - self.generated_at >= BATCH_TTL_S and not self.needs_write:
            self.needs_write = True
            self.write_reason = "expired"
        return self.needs_write

    # -- inputs -----------------------------------------------------------

    def _bucket(self, inputs: dict) -> PresenceProfile:
        labels = {
            str(k).lower(): float(v)
            for k, v in (inputs.get("mood_labels") or {}).items()
        }
        sad = max(labels.get("sad", 0.0), labels.get("suppressed", 0.0),
                  labels.get("wronged", 0.0), labels.get("guilty", 0.0))
        doubt = labels.get("doubtful", 0.0)
        bright = max(labels.get("happy", 0.0), labels.get("relieved", 0.0),
                     labels.get("joy", 0.0), labels.get("excited", 0.0))
        intensity = float(inputs.get("mood_intensity") or 0.0)
        trust = float(inputs.get("trust") or 0.5)
        closeness = float(inputs.get("closeness") or 0.2)
        friction = float(inputs.get("friction") or 0.0)

        tension = max(friction, 0.0) * 0.7 + sad * 0.35 + doubt * 0.15
        warmth = closeness * 0.5 + trust * 0.35 + bright * 0.25 - tension * 0.3
        if inputs.get("has_diary"):
            # Having shared days behind her warms the idle behaviour a little.
            warmth += 0.05
        energy = self.base_energy * (0.75 + 0.5 * max(intensity, 0.15))
        mouth = 0.15 + bright * 0.4 + max(sad - 0.4, 0.0) * 0.3

        # Hysteresis: once a bucket is active it takes a wider margin to leave
        # it. Without this, values sitting on a threshold (bright 0.45) flip the
        # bucket every tick, which resets the cue clock and makes the model fire
        # once a second.
        prev = self.profile.key
        bright_gate = 0.40 if prev == "bright" else 0.45
        sad_gate = 0.30 if prev == "low" else 0.35
        tension_gate = 0.40 if prev == "tense" else 0.45
        warmth_gate = 0.50 if prev == "cozy" else 0.55

        if tension >= tension_gate:
            key = "tense"
            mouth *= 0.2
        elif sad >= sad_gate and bright < 0.3:
            key = "low"
        elif bright >= bright_gate:
            key = "bright"
        elif warmth >= warmth_gate:
            key = "cozy"
        else:
            key = "calm"
        note = (
            f"tension={tension:.2f} warmth={warmth:.2f} "
            f"sad={sad:.2f} bright={bright:.2f}"
        )
        return PresenceProfile(
            key=key,
            energy=min(1.0, max(0.2, energy)),
            warmth=min(1.0, max(0.0, warmth)),
            tension=min(1.0, max(0.0, tension)),
            mouth=min(1.0, max(0.0, mouth)),
            note=note,
        )

    # -- scheduling -------------------------------------------------------

    def observe(self, inputs: dict) -> PresenceProfile:
        """Update the profile; a bucket change asks for a fresh batch."""
        previous = self.profile.key
        self.profile = self._bucket(inputs)
        if self.profile.key != previous:
            self._next_at = 0.0
            if self.generated_at > 0 or self.library:
                if not self.needs_write or self.write_reason == "profile_change":
                    self.needs_write = True
                    self.write_reason = "profile_change"
        return self.profile

    def _pool(self) -> dict[str, dict[str, float]]:
        if self.profile.key in self.library:
            return self.library[self.profile.key]
        return {
            kind: {"w": weight, "i": 0.45 + 0.35 * min(1.0, self.profile.energy)}
            for kind, weight in POOLS.get(self.profile.key, POOLS["calm"]).items()
        }

    def pool_kinds(self) -> list[str]:
        """Kinds the model may choose from right now (same pool as _pick)."""
        return list(self._pool().keys())

    def recent(self) -> list[str]:
        """The last few kinds, so the model can avoid repeating itself."""
        return list(self._last[-3:])

    def note(self, kind: str) -> None:
        """Remember a kind the model picked, so recent stays truthful."""
        self._last = (self._last + [kind])[-4:]

    def _pick(self, now: float) -> tuple[str, float] | None:
        pool = self._pool()
        # Mouth kinds are excluded entirely when the mouth appetite is low.
        candidates = {
            kind: value
            for kind, value in pool.items()
            if kind not in MOUTH_KINDS or self.profile.mouth >= 0.2
        }
        if not candidates:
            candidates = {
                kind: value for kind, value in pool.items() if kind not in MOUTH_KINDS
            }
        if not candidates:
            return None
        # Avoid the last two choices so it does not look like a loop.
        for kind in self._last[-2:]:
            candidates.pop(kind, None)
        if not candidates:
            return None
        kinds = list(candidates)
        weights = [candidates[kind]["w"] for kind in kinds]
        kind = self.rng.choices(kinds, weights=weights, k=1)[0]
        self._last = (self._last + [kind])[-4:]
        intensity = candidates[kind].get("i") or 0.45 + 0.35 * min(1.0, self.profile.energy)
        return kind, round(min(1.2, max(0.1, intensity)), 3)

    def next_cue(
        self, now: float, *, speaking: bool, silence_s: float = 0.0
    ) -> tuple[str, float] | None:
        """Return (kind, intensity) when it is time for an idle cue."""
        if speaking:
            return None
        if now < self._next_at:
            return None
        self._next_at = now + next_gap(silence_s, self.rng)
        if silence_s >= 300 and self.rng.random() < 0.4:
            # After a long silence she looks over once in a while instead of
            # fidgeting in place.
            pool = self._pool()
            for kind in ("idle_look_user", "idle_glance"):
                if kind in pool and kind not in self._last[-2:]:
                    self._last = (self._last + [kind])[-4:]
                    return kind, round(float(pool[kind].get("i", 0.5)), 3)
        return self._pick(now)


def idle_mouth_allowed(state: str, timeline_active: bool) -> bool:
    """Idle mouth cues are only safe when nothing else is using the mouth."""
    return state != "speaking" and not timeline_active


def silence_seconds(now: float, last_voice: float) -> float:
    return max(0.0, now - last_voice) if last_voice else 0.0


def quiet_bucket(seconds: float) -> str:
    """After a long silence she settles down instead of fidgeting."""
    if seconds >= 300:
        return "very_quiet"
    if seconds >= 90:
        return "quiet"
    return "normal"


def next_gap(seconds: float, rng: random.Random) -> float:
    bucket = quiet_bucket(seconds)
    if bucket == "very_quiet":
        return rng.uniform(MAX_GAP * 1.5, MAX_GAP * 2.5)
    if bucket == "quiet":
        return rng.uniform(MAX_GAP, MAX_GAP * 1.6)
    return rng.uniform(MIN_GAP, MAX_GAP)
