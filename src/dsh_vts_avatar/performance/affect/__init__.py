"""Affect data model and expression composer (Phases 1-2 of the emotion refactor).

Evidence sources describe a state; the composer turns that state into face and
head targets. The mixer remains the only writer to VTS.
"""

from .composer import (
    FACE_KEYS,
    HEAD_KEYS,
    NEUTRAL_FACE,
    NEUTRAL_HEAD,
    blend_face,
    blend_head,
    compose_face,
    compose_head,
)
from .fusion import SOURCE_WEIGHTS, AffectFusion, FusionSettings
from .log import AffectLog, read_log
from .presets import (
    ANCHORS,
    LABEL_ALIASES,
    LEGACY_LABELS,
    UNKNOWN_ANCHOR,
    anchor,
    canonical,
    is_known,
)
from .schema import (
    NEUTRAL_LABELS,
    SOURCES,
    AffectEvidence,
    AffectState,
    derive_valence_arousal,
    normalize_labels,
)

__all__ = [
    "ANCHORS",
    "AffectEvidence",
    "AffectFusion",
    "AffectLog",
    "AffectState",
    "FACE_KEYS",
    "FusionSettings",
    "HEAD_KEYS",
    "LABEL_ALIASES",
    "LEGACY_LABELS",
    "NEUTRAL_FACE",
    "NEUTRAL_HEAD",
    "NEUTRAL_LABELS",
    "SOURCES",
    "SOURCE_WEIGHTS",
    "UNKNOWN_ANCHOR",
    "anchor",
    "blend_face",
    "blend_head",
    "canonical",
    "compose_face",
    "compose_head",
    "derive_valence_arousal",
    "is_known",
    "normalize_labels",
    "read_log",
]
