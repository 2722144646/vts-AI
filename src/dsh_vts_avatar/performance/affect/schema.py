"""Affect data model: evidence in, one continuous state out.

Phase 1 only defines the vocabulary and the derivation. Nothing here writes to
VTS; the mixer keeps using the legacy preset path until the composer lands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time

from .presets import anchor, canonical

SOURCES = ("manual", "tool", "tag", "context", "classifier", "keyword", "ndi")

# What "neutral" looks like when no evidence ever arrived. It is deliberately
# weak: an empty state must read as "no expression", not as a calm actor.
NEUTRAL_LABELS: dict[str, float] = {"neutral": 1.0}


def normalize_labels(labels: dict[str, float] | None) -> dict[str, float]:
    """Clamp to 0..1, drop non-positive weights, and renormalize to sum 1."""
    clean: dict[str, float] = {}
    for raw_name, raw_weight in (labels or {}).items():
        name = canonical(raw_name)
        if not name:
            continue
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError):
            continue
        if weight <= 0.0:
            continue
        clean[name] = clean.get(name, 0.0) + min(1.0, weight)
    total = sum(clean.values())
    if total <= 0.0:
        return dict(NEUTRAL_LABELS)
    return {name: weight / total for name, weight in clean.items()}


def derive_valence_arousal(labels: dict[str, float]) -> tuple[float, float]:
    """Weights + anchors -> (valence, arousal). Neutral-only maps to (0, 0.15)."""
    valence = 0.0
    arousal = 0.0
    for name, weight in labels.items():
        v, a = anchor(name)
        valence += weight * v
        arousal += weight * a
    return _clamp(valence, -1.0, 1.0), _clamp(arousal, 0.0, 1.0)


@dataclass(slots=True)
class AffectEvidence:
    """One source's opinion. Never commands the face directly."""

    source: str
    labels: dict[str, float] = field(default_factory=dict)
    intensity: float = 0.7
    confidence: float = 0.8
    note: str = ""
    turn_id: str = ""
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise ValueError(f"unknown affect source: {self.source!r}")
        self.labels = normalize_labels(self.labels)
        self.intensity = _clamp(float(self.intensity), 0.0, 1.0)
        self.confidence = _clamp(float(self.confidence), 0.0, 1.0)

    @property
    def is_weak(self) -> bool:
        return self.confidence < 0.35

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "labels": dict(self.labels),
            "intensity": self.intensity,
            "confidence": self.confidence,
            "note": self.note,
            "turn_id": self.turn_id,
            "timestamp": self.timestamp,
        }


@dataclass(slots=True)
class AffectState:
    """The single continuous state every consumer reads."""

    labels: dict[str, float] = field(default_factory=lambda: dict(NEUTRAL_LABELS))
    intensity: float = 0.0
    confidence: float = 0.0
    valence: float = 0.0
    arousal: float = 0.0
    hold_until: float = 0.0
    source: str = "fusion"
    updated_at: float = field(default_factory=time.time)

    @classmethod
    def from_evidence(cls, evidence: AffectEvidence, *, source: str = "evidence") -> "AffectState":
        labels = normalize_labels(evidence.labels)
        valence, arousal = derive_valence_arousal(labels)
        return cls(
            labels=labels,
            intensity=evidence.intensity,
            confidence=evidence.confidence,
            valence=valence,
            arousal=arousal,
            hold_until=0.0,
            source=source,
        )

    @classmethod
    def neutral(cls) -> "AffectState":
        return cls()

    def is_neutral(self, threshold: float = 0.6) -> bool:
        return self.labels.get("neutral", 0.0) >= threshold

    def dominant(self, count: int = 1) -> list[tuple[str, float]]:
        ordered = sorted(self.labels.items(), key=lambda item: item[1], reverse=True)
        return ordered[:count]

    def to_dict(self) -> dict:
        return {
            "labels": dict(self.labels),
            "intensity": self.intensity,
            "confidence": self.confidence,
            "valence": self.valence,
            "arousal": self.arousal,
            "hold_until": self.hold_until,
            "source": self.source,
            "updated_at": self.updated_at,
        }


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
