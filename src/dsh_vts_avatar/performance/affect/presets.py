"""Affect anchors: the vocabulary shared by evidence sources and the composer.

An anchor is the *coordinate* of a label, not an expression. Expressions are
built in the composer (Phase 2) from a blend of labels; preset screenshots stay
as reference points for calibration and blind tests.

Values mirror ``performance.emotion.EmotionState.set`` so that the legacy path
and the affect path agree on what "happy" or "sad" means at intensity 1.0.
"""

from __future__ import annotations

# label -> (valence -1..1, arousal 0..1)
ANCHORS: dict[str, tuple[float, float]] = {
    "happy": (0.75, 0.55),
    "joy": (0.80, 0.60),
    "excited": (0.70, 0.90),
    "sad": (-0.70, 0.20),
    "tired": (-0.30, 0.10),
    "angry": (-0.65, 0.85),
    "surprised": (0.15, 0.90),
    "thinking": (-0.05, 0.35),
    "wronged": (-0.35, 0.35),
    "curious": (0.25, 0.50),
    "shy": (0.20, 0.35),
    "neutral": (0.00, 0.15),
    "calm": (0.15, 0.15),
    "suppressed": (-0.45, 0.20),
    "doubtful": (-0.10, 0.35),
    "guilty": (-0.35, 0.30),
    "relieved": (0.35, 0.20),
}

# Unknown labels must not silently become a strong emotion. They fall back to
# the neutral anchor, which only nudges the state.
UNKNOWN_ANCHOR = (0.0, 0.25)

# Labels the legacy keyword/tag path can emit. Kept for the compatibility
# layer that turns legacy presets into evidence instead of hard overrides.
LEGACY_LABELS = tuple(ANCHORS)


def anchor(label: str) -> tuple[float, float]:
    return ANCHORS.get(canonical(label), UNKNOWN_ANCHOR)


def is_known(label: str) -> bool:
    return canonical(label) in ANCHORS


# Chinese labels arrive from the manual tool / profile expression_map. Map them
# onto the same anchors instead of silently falling back to neutral.
LABEL_ALIASES: dict[str, str] = {
    "开心": "happy",
    "快乐": "happy",
    "高兴": "happy",
    "难过": "sad",
    "悲伤": "sad",
    "委屈": "wronged",
    "生气": "angry",
    "愤怒": "angry",
    "惊讶": "surprised",
    "吃惊": "surprised",
    "思考": "thinking",
    "害羞": "shy",
    "好奇": "curious",
    "累": "tired",
    "疲惫": "tired",
    "困": "tired",
    "中性": "neutral",
    "平静": "calm",
    "压抑": "suppressed",
    "压着": "suppressed",
    "忍着": "suppressed",
    "半信半疑": "doubtful",
    "怀疑": "doubtful",
    "心虚": "guilty",
    "内疚": "guilty",
    "愧疚": "guilty",
    "松了口气": "relieved",
    "释然": "relieved",
    "放心": "relieved",
}


def canonical(label: str) -> str:
    name = str(label or "").strip()
    lowered = name.lower()
    return LABEL_ALIASES.get(name) or LABEL_ALIASES.get(lowered) or lowered
