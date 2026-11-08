"""Continuous emotion state with attack/release and decay."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
import time


@dataclass
class EmotionState:
    valence: float = 0.0  # -1..1
    arousal: float = 0.0  # 0..1
    intensity: float = 0.0  # 0..1
    target_valence: float = 0.0
    target_arousal: float = 0.0
    target_intensity: float = 0.0
    preset: str = "neutral"
    preset_intensity: float = 0.0
    target_preset_intensity: float = 0.0
    last_update: float = 0.0
    hold_until: float = 0.0

    def __post_init__(self) -> None:
        if not self.last_update:
            self.last_update = time.perf_counter()

    def set(self, name: str, intensity: float = 0.7) -> None:
        name = name.lower().strip()
        intensity = max(0.0, min(1.0, float(intensity)))
        presets = {
            "happy": (0.75, 0.55),
            "joy": (0.8, 0.6),
            "excited": (0.7, 0.9),
            "sad": (-0.7, 0.2),
            "angry": (-0.65, 0.85),
            "surprised": (0.15, 0.9),
            "surprise": (0.15, 0.9),
            "thinking": (-0.05, 0.35),
            "wronged": (-0.35, 0.35),
            "aggrieved": (-0.35, 0.35),
            "curious": (0.25, 0.5),
            "shy": (0.2, 0.35),
            "neutral": (0.0, 0.15),
            "calm": (0.15, 0.15),
        }
        v, a = presets.get(name, (0.0, 0.25))
        self.target_valence = v * intensity
        self.target_arousal = a * intensity
        self.target_intensity = intensity
        self.preset = name
        self.target_preset_intensity = intensity
        # Do not jump the current value; let update() ease into the target.
        # A tiny immediate nudge keeps the face from feeling one beat late.
        self.valence += (self.target_valence - self.valence) * 0.12
        self.arousal += (self.target_arousal - self.arousal) * 0.12
        self.intensity += (self.target_intensity - self.intensity) * 0.12
        self.preset_intensity += (
            self.target_preset_intensity - self.preset_intensity
        ) * 0.12
        self.hold_until = time.perf_counter() + 1.8

    def update(self, dt: float) -> None:
        dt = max(0.0, min(0.2, dt))
        # Slow attack/release: expressions should bloom, hold, then fade.
        attack = 1.0 - math.exp(-dt / 0.38)
        release = 1.0 - math.exp(-dt / 2.2)
        self.valence += (self.target_valence - self.valence) * (attack if abs(self.target_valence) > abs(self.valence) else release)
        self.arousal += (self.target_arousal - self.arousal) * (attack if self.target_arousal > self.arousal else release)
        self.intensity += (self.target_intensity - self.intensity) * release
        self.preset_intensity += (
            self.target_preset_intensity - self.preset_intensity
        ) * (attack if self.target_preset_intensity > self.preset_intensity else release)
        # Hold the expression, then drift back toward a calm baseline.
        if time.perf_counter() >= self.hold_until:
            decay = 1.0 - math.exp(-dt / 4.0)
            self.target_valence *= 1.0 - decay * 0.30
            self.target_arousal *= 1.0 - decay * 0.22
            self.target_intensity *= 1.0 - decay * 0.18
            self.target_preset_intensity *= 1.0 - decay * 0.24
        self.last_update = time.perf_counter()


# Distinct, model-agnostic expression targets. The mixer blends these over the
# continuous valence/arousal baseline; this is what makes "sad" different from
# "low-energy happy" and "surprised" different from "excited".
EXPRESSION_TARGETS: dict[str, dict[str, float]] = {
    "happy": {"smile": 0.95, "eye": 0.90, "brow": 0.30, "angry": 0.0, "inner_brow": 0.04, "open": 0.04},
    "joy": {"smile": 0.99, "eye": 0.94, "brow": 0.35, "angry": 0.0, "inner_brow": 0.03, "open": 0.06},
    "excited": {"smile": 0.97, "eye": 0.96, "brow": 0.55, "angry": 0.0, "inner_brow": 0.03, "open": 0.18},
    "sad": {"smile": 0.0, "eye": 0.38, "brow": 0.02, "angry": 0.0, "inner_brow": 1.0, "open": 0.0, "gaze_x": 0.0, "gaze_y": -0.50, "brow_l": 0.12, "brow_r": 0.12},
    "angry": {"smile": 0.0, "eye": 0.50, "brow": 0.10, "angry": 0.60, "inner_brow": 0.50, "open": 0.02, "gaze_x": -0.25, "gaze_y": 0.10},
    "surprised": {"smile": 0.16, "eye": 1.0, "brow": 0.96, "angry": 0.0, "inner_brow": 0.08, "open": 0.70},
    "surprise": {"smile": 0.16, "eye": 1.0, "brow": 0.96, "angry": 0.0, "inner_brow": 0.08, "open": 0.70},
    "thinking": {"smile": 0.12, "eye": 0.98, "brow": 0.45, "angry": 0.0, "inner_brow": 0.08, "open": 0.02, "gaze_x": 0.85, "gaze_y": 0.55, "brow_l": 0.72, "brow_r": 0.20},
    "wronged": {"smile": 0.06, "eye": 0.58, "brow": 0.04, "angry": 0.0, "inner_brow": 0.82, "open": 0.02, "gaze_x": 0.65, "gaze_y": 0.35, "brow_l": 0.70, "brow_r": 0.20},
    "curious": {"smile": 0.24, "eye": 0.94, "brow": 0.52, "angry": 0.0, "inner_brow": 0.10, "open": 0.08},
    "shy": {"smile": 0.75, "eye": 0.38, "brow": 0.14, "angry": 0.0, "inner_brow": 0.40, "open": 0.02, "gaze_x": -0.50, "gaze_y": 0.55, "brow_l": 0.24, "brow_r": 0.24},
    "neutral": {"smile": 0.14, "eye": 0.93, "brow": 0.10, "angry": 0.0, "inner_brow": 0.04, "open": 0.02},
    "calm": {"smile": 0.20, "eye": 0.90, "brow": 0.08, "angry": 0.0, "inner_brow": 0.06, "open": 0.02},
    # --- nuanced L2 states (2026-11-07) ---------------------------------
    # suppressed: 难过的“收着”版，但这次做明显一些：嘴角压平、眼半闭、内眉提、低头。
    "suppressed": {"smile": 0.10, "eye": 0.34, "brow": 0.10, "angry": 0.0, "inner_brow": 0.75,
                   "open": 0.0, "gaze_x": 0.10, "gaze_y": -0.28, "brow_l": 0.12, "brow_r": 0.12},
    # doubtful: 半信半疑，夸张版：一边眉挑很高、一边压平，眼眯、头偏、视线斜。
    "doubtful": {"smile": 0.28, "eye": 0.64, "brow": 0.65, "angry": 0.0, "inner_brow": 0.28,
                 "open": 0.02, "gaze_x": 0.40, "gaze_y": 0.08, "brow_l": 0.75, "brow_r": 0.02},
    # guilty: 心虚，夸张版：低头躲视线、内眉高提、嘴角收紧。
    "guilty": {"smile": 0.12, "eye": 0.38, "brow": 0.12, "angry": 0.0, "inner_brow": 0.85,
               "open": 0.0, "gaze_x": 0.45, "gaze_y": -0.25, "brow_l": 0.12, "brow_r": 0.12},
    # relieved: 不再自带笑；笑交给哈气动作，表情只留一点点放松。
    "relieved": {"smile": 0.15, "eye": 0.52, "brow": 0.575, "angry": 0.0, "inner_brow": 0.0,
                 "open": 0.0, "gaze_x": 0.0, "gaze_y": 0.06},
}

EXPRESSION_HEAD: dict[str, dict[str, float]] = {
    "happy": {"yaw": 0.0, "pitch": -0.6, "roll": 1.2},
    "joy": {"yaw": 0.0, "pitch": -0.8, "roll": 1.8},
    "excited": {"yaw": 0.0, "pitch": -0.4, "roll": 2.4},
    "sad": {"yaw": 0.0, "pitch": -7.0, "roll": -3.5},
    "angry": {"yaw": 0.0, "pitch": 3.0, "roll": -6.0},
    "surprised": {"yaw": 0.0, "pitch": -5.0, "roll": 0.0},
    "thinking": {"yaw": 0.0, "pitch": 1.0, "roll": -10.0},
    "wronged": {"yaw": 3.0, "pitch": -2.0, "roll": 7.0},
    "curious": {"yaw": 1.2, "pitch": -0.6, "roll": 3.0},
    "shy": {"yaw": -2.0, "pitch": -10.0, "roll": 5.0},
    "neutral": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
    "calm": {"yaw": 0.0, "pitch": 0.0, "roll": 0.4},
    "suppressed": {"yaw": 0.0, "pitch": 4.0, "roll": -1.5},
    "doubtful": {"yaw": 3.0, "pitch": -1.5, "roll": 4.5},
    "guilty": {"yaw": -5.0, "pitch": 3.0, "roll": -1.0},
    "relieved": {"yaw": 0.0, "pitch": -2.0, "roll": 1.0},
}


_LEXICON: dict[str, tuple[str, float]] = {
    "太好了": ("happy", 0.8),
    "太棒": ("happy", 0.85),
    "好耶": ("happy", 0.8),
    "真好": ("happy", 0.6),
    "开心": ("happy", 0.8),
    "高兴": ("happy", 0.7),
    "哈哈": ("happy", 0.7),
    "喜欢": ("happy", 0.6),
    "可爱": ("happy", 0.6),
    "有趣": ("happy", 0.55),
    "厉害": ("happy", 0.6),
    "不错": ("happy", 0.5),
    "棒": ("happy", 0.55),
    "期待": ("happy", 0.55),
    "放心": ("happy", 0.45),
    "没问题": ("happy", 0.45),
    "当然": ("happy", 0.4),
    "谢谢": ("happy", 0.5),
    "好的": ("happy", 0.35),
    "收到": ("happy", 0.35),
    "抱歉": ("sad", 0.6),
    "对不起": ("sad", 0.6),
    "难过": ("sad", 0.8),
    "可惜": ("sad", 0.6),
    "唉": ("sad", 0.5),
    "累": ("sad", 0.5),
    "困": ("sad", 0.4),
    "生气": ("angry", 0.8),
    "讨厌": ("angry", 0.7),
    "烦": ("angry", 0.6),
    "离谱": ("angry", 0.6),
    "居然": ("surprised", 0.7),
    "竟然": ("surprised", 0.7),
    "真的吗": ("surprised", 0.8),
    "不会吧": ("surprised", 0.8),
    "天哪": ("surprised", 0.8),
    "哇": ("surprised", 0.8),
    "诶": ("surprised", 0.55),
    "咦": ("surprised", 0.55),
    "让我想想": ("thinking", 0.6),
    "也许": ("thinking", 0.4),
    "可能": ("thinking", 0.35),
    "嗯": ("thinking", 0.3),
    "great": ("happy", 0.8),
    "happy": ("happy", 0.8),
    "love": ("happy", 0.7),
    "nice": ("happy", 0.6),
    "sorry": ("sad", 0.7),
    "sad": ("sad", 0.7),
    "angry": ("angry", 0.8),
    "wow": ("surprised", 0.8),
    "really": ("surprised", 0.6),
    "maybe": ("thinking", 0.4),
}

_TAG_RE = re.compile(r"\[(emotion|情绪)\s*:\s*([a-zA-Z_\u4e00-\u9fff]+)(?:\s+intensity\s*[:=]\s*([0-9.]+))?\]", re.I)


_NEGATIONS = "不没别无非未"
_INTENSIFIERS = {
    "非常": 0.16,
    "特别": 0.14,
    "超级": 0.16,
    "太": 0.12,
    "很": 0.10,
    "好": 0.08,
    "有点": -0.16,
    "稍微": -0.14,
    "一点": -0.12,
}
_NEGATED = {
    "happy": ("sad", 0.48),
    "joy": ("sad", 0.48),
    "excited": ("calm", 0.40),
    "sad": ("calm", 0.40),
    "angry": ("calm", 0.42),
    "surprised": ("neutral", 0.40),
    "thinking": ("neutral", 0.38),
    "curious": ("neutral", 0.38),
}


def emotion_from_text(text: str) -> tuple[str, float] | None:
    """Cheap, local affect heuristic; no LLM round-trip and no visible tags.

    It searches all lexicon hits rather than returning the first one, handles
    common Chinese negation ("不开心") and intensity adverbs, and keeps the
    explicit tag path as the highest-priority side channel.
    """
    tag = _TAG_RE.search(text)
    if tag:
        try:
            intensity = float(tag.group(3)) if tag.group(3) else 0.8
        except ValueError:
            intensity = 0.8
        return tag.group(2).lower(), max(0.0, min(1.0, intensity))

    lowered = text.lower()
    best: tuple[str, float] | None = None
    best_score = 0.0
    for token, (preset, base_intensity) in _LEXICON.items():
        start = 0
        while True:
            index = lowered.find(token, start)
            if index < 0:
                break
            before = text[max(0, index - 3):index]
            intensity = base_intensity
            for word, delta in _INTENSIFIERS.items():
                if word in before:
                    intensity += delta
                    break
            if any(ch in before for ch in _NEGATIONS):
                preset, intensity = _NEGATED.get(preset, ("neutral", 0.4))
            intensity = max(0.05, min(1.0, intensity))
            if intensity > best_score:
                best = (preset, intensity)
                best_score = intensity
            start = index + max(1, len(token))

    if best is not None:
        return best
    if "?" in text or "？" in text:
        return ("curious", 0.45)
    if "!" in text or "！" in text:
        return ("excited", 0.55)
    if "~" in text or "～" in text:
        return ("happy", 0.45)
    if "..." in text or "……" in text:
        return ("thinking", 0.4)
    return None


def strip_emotion_tags(text: str) -> str:
    return _TAG_RE.sub("", text)
