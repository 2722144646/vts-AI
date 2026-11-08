"""Expression Composer: blended affect state -> face and head targets.

This is the Phase 2 piece. It does not write to VTS; it produces the same
target dictionary the legacy preset path produced, so the mixer can switch
between them without touching the mouth clock.

Design notes:

- The neutral baseline is *not* ``EXPRESSION_TARGETS["neutral"]``. That row
  is a preset screenshot anchor (smile 0.14 would read as a frown); the model's
  neutral input values are 0.50 / 0.55 / 0.575. The mixer used those values
  directly, so the composer keeps them as the base.
- A label missing from the table (e.g. ``tired``) contributes the neutral
  baseline for that key instead of zero, so mixing never drags the face to 0.
- Intensity scales the whole blend around the neutral base: 0 = neutral face,
  1 = the full blended anchor. This is what keeps a weak reading from looking
  like a strong one.
"""

from __future__ import annotations

from ..emotion import EXPRESSION_HEAD, EXPRESSION_TARGETS
from .presets import canonical
from .schema import AffectState

# The model's neutral input values. Sending 0 makes this avatar frown and
# collapse the brows, so 0 is wrong even when "no emotion" is intended.
NEUTRAL_FACE: dict[str, float] = {
    "smile": 0.50,  # 默认值，用户 2026-11-07 要求恢复，不要压平
    "eye": 0.55,
    "brow": 0.575,
    "angry": 0.0,
    "inner_brow": 0.0,
    "open": 0.0,
    "gaze_x": 0.0,
    "gaze_y": 0.0,
    "brow_l": 0.0,
    "brow_r": 0.0,
}

NEUTRAL_HEAD: dict[str, float] = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

FACE_KEYS = tuple(NEUTRAL_FACE)
HEAD_KEYS = tuple(NEUTRAL_HEAD)


def _label_face(label: str) -> dict[str, float]:
    name = canonical(label)
    # "neutral"/"calm" in EXPRESSION_TARGETS are *preset screenshot* anchors
    # (smile 0.14 would read as a frown). For blending, neutral means the
    # model's real neutral input values.
    if name in ("neutral", "calm"):
        return dict(NEUTRAL_FACE)
    row = EXPRESSION_TARGETS.get(name)
    if not row:
        return dict(NEUTRAL_FACE)
    return {key: float(row.get(key, NEUTRAL_FACE[key])) for key in FACE_KEYS}


def _label_head(label: str) -> dict[str, float]:
    row = EXPRESSION_HEAD.get(canonical(label))
    if not row:
        return dict(NEUTRAL_HEAD)
    return {key: float(row.get(key, 0.0)) for key in HEAD_KEYS}


def blend_face(labels: dict[str, float]) -> dict[str, float]:
    """Weighted average of each label's face anchor; weights are assumed to sum to 1."""
    out = {key: 0.0 for key in FACE_KEYS}
    for label, weight in labels.items():
        row = _label_face(label)
        for key in FACE_KEYS:
            out[key] += weight * row[key]
    return out


def blend_head(labels: dict[str, float]) -> dict[str, float]:
    out = {key: 0.0 for key in HEAD_KEYS}
    for label, weight in labels.items():
        row = _label_head(label)
        for key in HEAD_KEYS:
            out[key] += weight * row[key]
    return out


def compose_face(state: AffectState, gain: float = 1.0) -> dict[str, float]:
    """Blend labels, then scale toward neutral by intensity * gain."""
    blended = blend_face(state.labels)
    strength = _clamp(state.intensity * gain, 0.0, 1.0)
    return {
        key: NEUTRAL_FACE[key] * (1.0 - strength) + blended[key] * strength
        for key in FACE_KEYS
    }


def compose_head(state: AffectState, gain: float = 1.0) -> dict[str, float]:
    """Blend head bias; the neutral pose contributes nothing, so no 1-strength mix."""
    blended = blend_head(state.labels)
    strength = _clamp(state.intensity * gain, 0.0, 1.0)
    return {key: blended[key] * strength for key in HEAD_KEYS}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
