"""Application-layer performance mixer.

VTS is a renderer here, not a mix bus. Every parameter has one owner and a
composition rule; only the final values are sent in one `set` request.
"""

from __future__ import annotations

import math
import random
import time
from bisect import bisect_right
from dataclasses import dataclass, field

import numpy as np

from ..config import AvatarConfig
from ..tts.timeline import Timeline, VisemeSegment
from ..tts.visemes import VisemeSpec, map_phoneme
from ..vts.profile import ModelProfile
from .actions import ActionCue
from .affect.composer import compose_face, compose_head
from .affect.schema import AffectState
from .emotion import EXPRESSION_HEAD, EXPRESSION_TARGETS, EmotionState


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


# Diphthongs are one Piper segment, but the mouth should travel through both
# targets instead of holding a single shape. Values are (start, end) vowel
# phones; the shape crossfades over the second half of the segment.
DIPHTHONGS: dict[str, tuple[str, str]] = {
    "ai": ("a", "i"),
    "ei": ("e", "i"),
    "ao": ("a", "o"),
    "ou": ("o", "u"),
    "ia": ("i", "a"),
    "ie": ("i", "e"),
    "ua": ("u", "a"),
    "uo": ("u", "o"),
    "iao": ("i", "o"),
    "iou": ("i", "u"),
    "uai": ("u", "i"),
    "uei": ("u", "i"),
}


def smoothstep(x: float) -> float:
    x = clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _smooth(current: float, target: float, dt: float, attack: float, release: float) -> float:
    tau = attack if abs(target) > abs(current) else release
    alpha = 1.0 - math.exp(-max(0.0, dt) / max(0.001, tau))
    return current + (target - current) * alpha


class GazeController:
    """Slow, bounded idle gaze. Expression gaze overrides it."""

    def __init__(self, seed: int | None = None, energy: float = 0.55):
        rng = random.Random(seed)
        self._phase = [rng.uniform(0, math.tau) for _ in range(4)]
        self._start = time.perf_counter()
        # 有存在感 / 更活泼：more energy means a slightly wider and quicker
        # gaze wander. Kept bounded so it never reads as a scanning loop.
        self._amp = 0.75 + 0.60 * clamp(energy, 0.0, 1.0)
        self._speed = 0.85 + 0.50 * clamp(energy, 0.0, 1.0)

    def update(self, now: float) -> tuple[float, float]:
        t = max(0.0, now - self._start) * self._speed
        x = (0.20 * math.sin(t * 0.23 + self._phase[0]) + 0.06 * math.sin(t * 0.71 + self._phase[1])) * self._amp
        y = (0.11 * math.sin(t * 0.19 + self._phase[2]) + 0.04 * math.sin(t * 0.53 + self._phase[3])) * self._amp
        return clamp(x, -0.45, 0.45), clamp(y, -0.30, 0.30)


class BlinkController:
    """Self-owned blink. VTS auto-blink should be disabled for this to be clean."""

    def __init__(self, per_minute: float = 14.0, seed: int | None = None):
        self.per_minute = max(2.0, min(60.0, per_minute))
        self._rng = random.Random(seed)
        self._next = time.perf_counter() + self._interval()
        self._started: float | None = None

    def _interval(self) -> float:
        base = 60.0 / self.per_minute
        return base * self._rng.uniform(0.65, 1.45)

    def update(self, now: float) -> float:
        if self._started is None and now >= self._next:
            self._started = now
            self._next = now + self._interval()
        if self._started is None:
            return 1.0
        t = (now - self._started) / 0.16
        if t >= 1.0:
            self._started = None
            return 1.0
        if t < 0.42:
            return 1.0 - smoothstep(t / 0.42)
        return smoothstep((t - 0.42) / 0.58)

    def reset(self) -> None:
        self._started = None
        self._next = time.perf_counter() + self._interval()


@dataclass
class PerformanceMixer:
    avatar: AvatarConfig
    sample_rate: int
    profile: ModelProfile | None = None
    backend: str = "continuous"
    emotion: EmotionState = field(default_factory=EmotionState)
    state: str = "idle"
    actions: list[ActionCue] = field(default_factory=list)
    # Phase 2: when enabled, face/head targets come from the composer instead
    # of the single-preset lookup. Legacy stays the default until the affect
    # path passes its own acceptance run.
    affect: AffectState | None = None
    use_affect: bool = False

    def __post_init__(self) -> None:
        self.backend = self.backend or "continuous"
        self._last_time = time.perf_counter()
        self._started = self._last_time
        self._last_sample = 0
        self._blink = BlinkController(self.avatar.blink_per_min + 2.0 * clamp(getattr(self.avatar, "idle_energy", 0.55), 0.0, 1.0))
        self._idle_energy = clamp(getattr(self.avatar, "idle_energy", 0.55), 0.0, 1.0)
        self._gaze = GazeController(energy=self._idle_energy)
        self._values: dict[str, float] = {}
        self._vel: dict[str, float] = {}
        self._state_bias = {"x": 0.0, "y": 0.0, "z": 0.0}
        self._speech_blend = 0.0
        self._speech_target = 0.0
        # Phase 2 baseline: slow mood layer under the transient affect.
        self.baseline = None
        self.baseline_weight = 0.0
        self._affect_stamp = 0.0
        self._last_turn_sample = -10**9
        self._mouth_backend = self._decide_backend()

    def _decide_backend(self) -> str:
        requested = (self.avatar.mouth_backend or "auto").lower()
        if requested != "auto":
            return requested
        if self.profile is not None:
            return self.profile.choose_mouth_backend(self.avatar, "auto")
        return "continuous"

    def set_state(self, state: str) -> None:
        state = (state or "idle").lower()
        if state not in {"idle", "thinking", "speaking", "listening", "error"}:
            state = "idle"
        self.state = state
        self._speech_target = 1.0 if state == "speaking" else 0.0
        if state == "speaking":
            # Idle motions yield the moment she starts talking; they must not
            # keep driving the mouth or neck under the speech timeline.
            self.actions = [cue for cue in self.actions if not cue.kind.startswith("idle_")]

    def set_profile(self, profile: ModelProfile | None, backend: str | None = None) -> None:
        self.profile = profile
        if backend is not None:
            self.backend = backend
        self._mouth_backend = backend or self._decide_backend()
        # Per-model tuning must win over global config. If a model has no
        # tuning entry yet, fall back to the global AvatarConfig value.
        self._blink.per_minute = self._gain("blink_per_min", self.avatar.blink_per_min)

    def _gain(self, name: str, default: float) -> float:
        if self.profile is not None:
            value = self.profile.tuning.get(name)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    pass
        return float(default)

    def set_emotion(self, name: str, intensity: float = 0.7) -> None:
        # If the user demonstrated a custom expression under a semantic name,
        # prefer that recording over the generic preset. This is the bridge
        # between the demo dataset and live expression selection.
        # Thinking/shy have no calibrated user take yet. Keep them on the
        # generic procedural presets so they can still be triggered and
        # improvised; do not silently replace them with an old learned take.
        alias = {
            "happy": "开心",
            "sad": "难过",
            "surprised": "惊讶",
            "wronged": "委屈",
        }.get((name or "").lower())
        if alias and self.profile is not None and alias in self.profile.expression_map:
            name = alias
        self.emotion.set(name, intensity)

    def set_affect(self, state: AffectState | None) -> None:
        """Attach the continuous affect state used by the composer path."""
        self.affect = state
        self._affect_stamp = time.time()

    def set_baseline(self, state: AffectState | None, weight: float = 0.0) -> None:
        """Attach the slow mood baseline; 0 disables it and restores old behavior."""
        self.baseline = state
        self.baseline_weight = max(0.0, min(1.0, float(weight)))

    #: How long a described feeling holds before it starts fading, and the
    #: release time constant. Evidence only arrives when someone speaks, so
    #: without a decay the last `feel()` would smile forever.
    AFFECT_HOLD_S = 2.0
    AFFECT_RELEASE_S = 2.2

    def _update_affect(self, dt: float) -> None:
        if not self.use_affect or self.affect is None:
            return
        if time.time() - self._affect_stamp <= self.AFFECT_HOLD_S:
            return
        self.affect.intensity *= math.exp(-max(0.0, dt) / self.AFFECT_RELEASE_S)

    def add_action(
        self,
        kind: str,
        intensity: float = 0.7,
        *,
        at_sample: int | None = None,
        duration_ms: int | None = None,
        curve: str = "minjerk",
    ) -> ActionCue | None:
        cue = ActionCue.create(
            kind,
            start=int(self._last_sample if at_sample is None else at_sample),
            sample_rate=self.sample_rate,
            intensity=intensity,
            duration_ms=duration_ms,
            curve=curve,
        )
        # Head channels are exclusive: a nod must not start inside a head turn
        # (or vice versa), otherwise the summed angles look like a pop.
        if cue.channel == "head":
            for active in self.actions:
                if active.channel != "head" or active.finished:
                    continue
                if active.start < cue.start + cue.duration and active.start + active.duration > cue.start:
                    return None
        if cue.kind in {"turn_look_left", "turn_look_right"}:
            # Do not schedule another look-away within 2s; the first one needs
            # time to read as a gesture, not a scanning loop.
            if cue.start - self._last_turn_sample < int(self.sample_rate * 2.0):
                return None
        self.actions.append(cue)
        if cue.kind in {"turn_look_left", "turn_look_right"}:
            self._last_turn_sample = cue.start
        if len(self.actions) > 24:
            self.actions = self.actions[-24:]
        return cue

    def reset(self) -> None:
        self.actions.clear()
        self._last_turn_sample = -10**9
        self._blink.reset()
        self._last_time = time.perf_counter()

    def compute(self, sample: int, timeline: Timeline | None, now: float | None = None) -> dict[str, float]:
        now = time.perf_counter() if now is None else now
        self._last_sample = int(sample)
        dt = clamp(now - self._last_time, 0.001, 0.15)
        self._last_time = now
        self.emotion.update(dt)
        self._update_affect(dt)
        self._update_state_bias(dt)
        self._speech_blend = _smooth(
            self._speech_blend, self._speech_target, dt, 0.22, 0.45
        )

        segment = timeline.at(sample) if timeline is not None else _silence()
        rms = timeline.rms_at(sample) if timeline is not None else 0.0
        env = _energy(rms)
        bilabial = (
            timeline.bilabial_at(sample, int(self.sample_rate * 0.020))
            if timeline is not None
            else None
        )
        mouth_segment = bilabial or segment

        speech = (
            self._speech(bilabial, env, int(sample))
            if bilabial is not None
            else self._speech_coarticulated(timeline, int(sample), segment, env)
        )
        emotional = self._emotion_targets()
        heads = self._head_targets(now)
        actions = self._action_targets(int(sample))
        blink = self._blink.update(now)

        smile = clamp(
            emotional["smile"] + speech["form"] * 0.45 + actions.get("smile", 0.0) * 1.0,
            -1.0,
            1.0,
        )
        mouth_open = max(speech["open"], emotional["open"], actions.get("open", 0.0))
        if self.state == "thinking":
            mouth_open *= 0.25
        voice = speech["voice"]

        # Blink is applied after smoothing (below): the 0.16s blink is faster
        # than the eye spring, so multiplying the target here only ever closed
        # the eyes part-way ("眨巴一下" without a real blink).
        eye_base = clamp(emotional["eye"], 0.0, 1.0)
        brow = emotional["brow"]
        angry = emotional["angry"]

        head_gain = self._gain("head_gain", self.avatar.head_gain)
        mapping = (
            self.profile.expression_map.get(self.emotion.preset)
            if self.profile is not None
            else None
        )
        mapping_strength = clamp(
            self.emotion.preset_intensity * self._gain("expression_gain", 1.0),
            0.0,
            1.0,
        )
        if mapping:
            # A model-specific map can add its own neck bias; this is what
            # makes "sad" bow the head instead of only lowering the eyelids.
            heads["x"] += mapping.get("FaceAngleX", 0.0) * mapping_strength
            heads["y"] += mapping.get("FaceAngleY", 0.0) * mapping_strength
            heads["z"] += mapping.get("FaceAngleZ", 0.0) * mapping_strength
            mouth_open = max(mouth_open, mapping.get("MouthOpen", 0.0) * mapping_strength)
        head_x = heads["x"] + actions.get("yaw", 0.0) * head_gain
        head_y = heads["y"] + actions.get("pitch", 0.0) * head_gain
        head_z = heads["z"] + actions.get("roll", 0.0) * head_gain
        if "eye_wide" in actions:
            eye_base = max(eye_base, clamp(actions["eye_wide"], 0.0, 1.0))

        body_x = math.sin((now - self._started) * 2 * math.pi * 0.06 + 1.4) * 0.9
        body_y = math.sin((now - self._started) * 2 * math.pi * 0.04 + 0.2) * 0.4
        body_z = math.sin((now - self._started) * 2 * math.pi * 0.05 + 2.7) * 1.2

        params: dict[str, float] = {}
        a = self.avatar
        mouth_gain = self._gain("mouth_gain", a.mouth_gain)
        jaw = 0.0
        funnel = 0.0
        pucker = 0.0
        phoneme = (segment.phoneme or "").lower()
        bilabial = phoneme in {"p", "b", "m"}
        if segment.kind == "vowel":
            if segment.viseme == "O":
                funnel = 0.65
                pucker = 0.55
            elif segment.viseme == "U":
                funnel = 0.30
                pucker = 0.80
            jaw = 0.75 * mouth_open
        elif segment.kind == "consonant":
            # Bilabials close the jaw; other consonants keep a small opening.
            jaw = 0.0 if segment.open <= 0.06 else 0.45 * mouth_open
            if phoneme in {"p", "b", "m"}:
                # 蒂亚 audit: MouthPressLipOpen at either polarity opens the
                # lips instead of closing them, so do not drive it here.
                jaw = 0.0
                mouth_open = min(mouth_open, 0.035)
        if self._mouth_backend in ("continuous", "hybrid"):
            params[a.mouth_open] = self._sv(
                a.mouth_open,
                mouth_open * mouth_gain,
                dt,
                0.010 if bilabial else (0.012 if mouth_segment.kind == "consonant" else 0.025),
                0.020 if bilabial else (0.035 if mouth_segment.kind == "consonant" else 0.080),
            )
        if self._mouth_backend in ("vowel", "hybrid"):
            # Always write all five voice params. Previously a consonant
            # returned an empty voice dict, so VTS kept the last vowel alive
            # through /p/ /b/ /m/ and the mouth no longer matched the text.
            for name in (a.voice_a, a.voice_i, a.voice_u, a.voice_e, a.voice_o):
                target = float(voice.get(name, 0.0)) * mouth_gain
                params[name] = self._sv(
                    name,
                    target,
                    dt,
                    0.010 if bilabial else (0.012 if mouth_segment.kind == "consonant" else 0.020),
                    0.020 if bilabial else (0.030 if mouth_segment.kind == "consonant" else 0.055),
                )
            # The vowel blendshape defines the shape; MouthOpen supplies the
            # visible jaw opening. Do not scale it down a second time after the
            # viseme curve has already been calibrated to the model ceiling.
            params[a.mouth_open] = self._sv(
                a.mouth_open,
                mouth_open * mouth_gain,
                dt,
                0.010 if bilabial else (0.012 if mouth_segment.kind == "consonant" else 0.025),
                0.020 if bilabial else (0.035 if mouth_segment.kind == "consonant" else 0.080),
            )
        if self.profile is not None:
            if self.profile.has(a.jaw_open):
                params[a.jaw_open] = self._sv(a.jaw_open, jaw * mouth_gain, dt, 0.025, 0.085)
            if self.profile.has(a.mouth_funnel):
                params[a.mouth_funnel] = self._sv(a.mouth_funnel, funnel * mouth_gain, dt, 0.030, 0.090)
            if self.profile.has(a.mouth_pucker):
                params[a.mouth_pucker] = self._sv(a.mouth_pucker, pucker * mouth_gain, dt, 0.030, 0.090)
            if self.profile.has(a.voice_silence):
                # VTS's automatic lip-sync setup gates the mouth layer with
                # ParamSilence. Keep it "not silent" for the whole spoken
                # utterance so the layer does not flicker between words.
                # 2026-11-07: an action that opens the mouth (e.g. sigh) must
                # also lift the gate, otherwise MouthOpen is ignored at idle.
                wants_mouth = actions.get("open", 0.0) > 0.01
                gate = 0.0 if (self.state == "speaking" or wants_mouth) else 1.0
                params[a.voice_silence] = self._sv(a.voice_silence, gate, dt, 0.18, 0.45)
        params[a.mouth_smile] = self._sv(a.mouth_smile, smile, dt, 0.18, 0.55)
        params[a.eye_left] = clamp(
            self._sv(a.eye_left, eye_base, dt, 0.07, 0.18) * blink, 0.0, 1.0
        )
        params[a.eye_right] = clamp(
            self._sv(a.eye_right, eye_base, dt, 0.07, 0.18) * blink, 0.0, 1.0
        )
        if (self.avatar.head_motion or "minjerk").lower() == "spring":
            head_smooth = lambda name, target: self._spring(
                name,
                target,
                dt,
                freq=self.avatar.head_spring_hz,
                zeta=self.avatar.head_spring_zeta,
            )
        elif (self.avatar.head_motion or "").lower() == "bezier":
            # The action curves themselves are cubic Bezier; the idle sway is
            # a smooth sine. A light one-pole only removes state/emotion steps,
            # it does not add a second spring or visible lag.
            head_smooth = lambda name, target: self._sv(name, target, dt, 0.08, 0.16)
        else:
            # Minimum-jerk keyframes plus a light asymmetric low-pass: no
            # spring bounce, no visible snap at the start of a nod.
            head_smooth = lambda name, target: self._sv(name, target, dt, 0.13, 0.30)
        params[a.head_x] = self._head_value(a.head_x, clamp(head_x, -30, 30), dt, head_smooth)
        params[a.head_y] = self._head_value(a.head_y, clamp(head_y, -30, 30), dt, head_smooth)
        params[a.head_z] = self._head_value(a.head_z, clamp(head_z, -30, 30), dt, head_smooth)
        # Body is a custom input in VTS; only send it when the contract proves
        # that it exists (or the user explicitly configured a custom contract).
        if self.profile is not None and self.profile.has(a.body_x):
            params[a.body_x] = self._spring(a.body_x, clamp(body_x, -30, 30), dt, freq=0.8)
            params[a.body_y] = self._spring(a.body_y, clamp(body_y, -30, 30), dt, freq=0.8)
            params[a.body_z] = self._spring(a.body_z, clamp(body_z, -30, 30), dt, freq=0.8)
        # Optional emotion parameters; harmless if absent because profile filters.
        params["Brows"] = self._sv("Brows", clamp(brow, -1, 1), dt, 0.18, 0.50)
        params["FaceAngry"] = self._sv("FaceAngry", clamp(angry, 0, 1), dt, 0.20, 0.60)
        # Looking aside is what makes "thinking" read as thinking instead of
        # sad. The values are expression targets already blended by the mixer.
        idle_gaze_x, idle_gaze_y = self._gaze.update(now)
        gaze_blend = 1.0 - clamp(
            self.emotion.preset_intensity * self._gain("expression_gain", 1.0), 0.0, 1.0
        ) * 0.85
        if self.state == "listening":
            # Keep eye contact while the other side is talking; do not drift
            # away as if the avatar were bored.
            gaze_blend *= 0.30
        for eye_name in ("EyeLeftX", "EyeRightX", "EyeLeftY", "EyeRightY"):
            if self.profile is not None and self.profile.has(eye_name):
                axis = "gaze_x" if eye_name.endswith("X") else "gaze_y"
                idle = idle_gaze_x if axis == "gaze_x" else idle_gaze_y
                target = (
                    emotional.get(axis, 0.0)
                    + idle * gaze_blend
                    + actions.get("gaze_x" if axis == "gaze_x" else "gaze_y", 0.0)
                )
                params[eye_name] = self._sv(eye_name, clamp(target, -1.0, 1.0), dt, 0.18, 0.40)
        for brow_name, key in (("BrowLeftY", "brow_l"), ("BrowRightY", "brow_r")):
            if self.profile is not None and self.profile.has(brow_name):
                params[brow_name] = self._sv(
                    brow_name,
                    clamp(
                        emotional.get(key, 0.0)
                        + actions.get("brow_l" if key == "brow_l" else "brow_r", 0.0),
                        0.0,
                        1.0,
                    ),
                    dt,
                    0.18,
                    0.45,
                )
        if self.profile is not None and self.profile.has(a.brow_inner):
            params[a.brow_inner] = self._sv(
                a.brow_inner,
                clamp(emotional["inner_brow"] + actions.get("inner_brow", 0.0), 0, 1),
                dt,
                0.18,
                0.55,
            )

        if mapping and mapping_strength > 0.001:
            # Direct per-model expression parameters. This is intentionally
            # applied after the generic template so 蒂亚 can use her own
            # brow/eye/head vocabulary while other models keep the fallback.
            direct = {
                a.mouth_smile: "MouthSmile",
                a.eye_left: "EyeOpenLeft",
                a.eye_right: "EyeOpenRight",
                "Brows": "Brows",
                "FaceAngry": "FaceAngry",
                "EyeLeftX": "EyeLeftX",
                "EyeRightX": "EyeRightX",
                "EyeLeftY": "EyeLeftY",
                "EyeRightY": "EyeRightY",
                "BrowLeftY": "BrowLeftY",
                "BrowRightY": "BrowRightY",
                "CheekPuff": "CheekPuff",
            }
            for param, key in direct.items():
                if key not in mapping:
                    continue
                if self.profile is not None and self.profile.input_params and not self.profile.has(key):
                    continue
                current = params.get(param, mapping[key])
                target = current * (1.0 - mapping_strength) + mapping[key] * mapping_strength
                params[param] = self._sv(param, target, dt, 0.18, 0.45)
            if (
                a.brow_inner in mapping
                and self.profile is not None
                and (not self.profile.input_params or self.profile.has(a.brow_inner))
            ):
                target = (
                    params.get(a.brow_inner, mapping[a.brow_inner]) * (1.0 - mapping_strength)
                    + mapping[a.brow_inner] * mapping_strength
                )
                params[a.brow_inner] = self._sv(a.brow_inner, target, dt, 0.18, 0.45)

        if self.profile is not None:
            params = self.profile.filter(params)
        return {key: float(value) for key, value in params.items() if value == value}

    # -- speech -----------------------------------------------------------

    @staticmethod
    def _blend_speech(a: dict[str, object], b: dict[str, object], weight: float) -> dict[str, object]:
        weight = clamp(weight, 0.0, 1.0)
        voice_a = a.get("voice") or {}
        voice_b = b.get("voice") or {}
        voice = {}
        for key in set(voice_a) | set(voice_b):
            voice[key] = float(voice_a.get(key, 0.0)) * (1.0 - weight) + float(voice_b.get(key, 0.0)) * weight
        return {
            "open": float(a.get("open", 0.0)) * (1.0 - weight) + float(b.get("open", 0.0)) * weight,
            "form": float(a.get("form", 0.0)) * (1.0 - weight) + float(b.get("form", 0.0)) * weight,
            "voice": voice,
            "viseme": a.get("viseme", "") if weight < 0.5 else b.get("viseme", ""),
        }

    def _speech_coarticulated(
        self,
        timeline: Timeline | None,
        sample: int,
        segment: VisemeSegment,
        env: float,
    ) -> dict[str, object]:
        """Blend 30ms of the previous shape and 20ms of the next shape.

        A hard switch at phoneme boundaries is what makes lips look like a
        slideshow. The viseme still owns the identity; neighbours only bleed
        into the transition.
        """
        base = self._speech(segment, env, sample)
        if timeline is None or len(timeline.segments) < 2:
            return base
        starts = getattr(timeline, "_starts", None)
        if not starts:
            return base
        index = bisect_right(starts, sample) - 1
        if index < 0 or index >= len(timeline.segments):
            return base
        current = timeline.segments[index]
        if current.start != segment.start or current.end != segment.end:
            return base
        if (current.phoneme or "").lower() in {"p", "b", "m"}:
            # Keep bilabial closures crisp; blending a neighbouring vowel into
            # a 10-30ms /p/ or /m/ makes it look like the mouth never closes.
            return base
        previous_seg = timeline.segments[index - 1] if index > 0 else None
        following_seg = (
            timeline.segments[index + 1]
            if index + 1 < len(timeline.segments)
            else None
        )
        vowel_vowel = (
            segment.kind == "vowel"
            and previous_seg is not None
            and previous_seg.kind == "vowel"
        )
        # Vowel→vowel needs a wider transition to avoid a visible step;
        # vowel→consonant closes faster so plosives still read as closed.
        pre = max(1, int(self.sample_rate * (0.040 if vowel_vowel else 0.025)))
        post = max(
            1,
            int(
                self.sample_rate
                * (0.030 if following_seg is not None and following_seg.kind == "vowel" else 0.018)
            ),
        )
        if previous_seg is not None and sample - current.start < pre:
            previous = self._speech(previous_seg, env, current.start)
            peak = 0.55 if vowel_vowel else (0.40 if segment.kind == "vowel" else 0.30)
            weight = peak * (1.0 - (sample - current.start) / pre)
            return self._blend_speech(previous, base, 1.0 - weight)
        if following_seg is not None and current.end - sample < post:
            following = self._speech(following_seg, env, current.end)
            peak = 0.55 if vowel_vowel else (0.40 if following_seg.kind == "vowel" else 0.30)
            weight = peak * (1.0 - (current.end - sample) / post)
            return self._blend_speech(base, following, weight)
        return base

    def _vowel_shape(self, spec: VisemeSpec, env: float) -> dict[str, object]:
        amp = 0.72 + 0.28 * env
        vowel = spec.vowel_gain * (0.65 + 0.35 * env)
        return {
            "open": spec.open * amp,
            "form": spec.form,
            "voice": {self._voice_param(spec.viseme): vowel},
            "viseme": spec.viseme,
        }

    def _speech(
        self, segment: VisemeSegment, env: float, sample: int | None = None
    ) -> dict[str, object]:
        if segment.kind == "silence":
            return {"open": 0.0, "form": 0.0, "voice": {}, "viseme": ""}
        if segment.kind == "consonant":
            open_value = segment.open
            # Bilabials must override RMS completely.
            if segment.open <= 0.06:
                open_value = 0.035
            voice = {}
            if segment.viseme:
                voice = {self._voice_param(segment.viseme): segment.vowel_gain * 0.35}
            return {"open": open_value, "form": 0.0, "voice": voice, "viseme": segment.viseme}
        pair = DIPHTHONGS.get((segment.phoneme or "").lower())
        if pair and sample is not None and segment.end > segment.start:
            progress = clamp(
                (sample - segment.start) / float(segment.end - segment.start), 0.0, 1.0
            )
            # Hold the first target briefly, then travel; a linear ramp from
            # the first millisecond reads as a twitch, not a diphthong.
            weight = smoothstep(clamp((progress - 0.30) / 0.55, 0.0, 1.0))
            first = self._vowel_shape(map_phoneme(pair[0], "zh"), env)
            second = self._vowel_shape(map_phoneme(pair[1], "zh"), env)
            return self._blend_speech(first, second, weight)
        # Vowel: RMS modulates amplitude, but the model's useful visual range
        # must be reached on normal speech; a low floor made every mouth look
        # half-open. The viseme openness already encodes the vowel identity.
        amp = 0.72 + 0.28 * env
        open_value = segment.open * amp
        vowel = segment.vowel_gain * (0.65 + 0.35 * env)
        voice = {self._voice_param(segment.viseme): vowel}
        return {"open": open_value, "form": segment.form, "voice": voice, "viseme": segment.viseme}

    def _voice_param(self, viseme: str) -> str:
        a = self.avatar
        return {
            "A": a.voice_a,
            "I": a.voice_i,
            "U": a.voice_u,
            "E": a.voice_e,
            "O": a.voice_o,
        }.get(viseme, a.voice_a)

    # -- emotion ----------------------------------------------------------

    def _emotion_targets(self) -> dict[str, float]:
        if self.use_affect and self.affect is not None:
            # Phase 2: multi-label blend; the neutral base and the intensity
            # ramp live in the composer, not here.
            face = compose_face(self.affect, self._gain("expression_gain", 1.0))
            if self.baseline is not None and self.baseline_weight > 0:
                base = compose_face(self.baseline, self._gain("expression_gain", 1.0))
                weight = self.baseline_weight
                for key in face:
                    face[key] = face[key] * (1.0 - weight) + base.get(key, 0.0) * weight
            return face
        if self.use_affect and self.baseline is not None and self.baseline_weight > 0:
            # Idle with a mood: the baseline alone keeps the face off neutral.
            return compose_face(
                self.baseline,
                self._gain("expression_gain", 1.0) * self.baseline_weight,
            )
        if self.emotion.preset == "neutral" or self.emotion.preset_intensity < 0.08:
            # The model maps MouthSmile through MouthForm -1..1 and Brows
            # through -1..1 with an input floor of 0.15. Sending 0 made the
            # mouth frown and the brows collapse; these are the model's neutral
            # input values, not zero.
            return {
                "smile": 0.50,
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
        v = self.emotion.valence
        a = self.emotion.arousal
        i = self.emotion.intensity
        smile = 0.20 + 0.82 * v * (0.6 + 0.4 * i)
        # Surprise widens; sadness lowers; anger narrows and lowers brow.
        eye = 0.92 + 0.10 * max(0.0, a - 0.55) - 0.12 * max(0.0, -v) - 0.04 * max(0.0, v)
        brow = 0.10 + 0.55 * a - 0.45 * max(0.0, -v)
        angry = 0.0
        if v < -0.15 and a > 0.55:
            angry = min(1.0, (-v) * (a - 0.4) * 1.5)
        inner_brow = max(0.0, -v) * 0.9
        open_emotion = 0.35 * max(0.0, a - 0.65) if v >= 0 else 0.0
        targets = {
            "smile": smile,
            "eye": eye,
            "brow": brow,
            "angry": angry,
            "inner_brow": inner_brow,
            "open": open_emotion,
            "gaze_x": 0.0,
            "gaze_y": 0.0,
            "brow_l": 0.0,
            "brow_r": 0.0,
        }
        expression = EXPRESSION_TARGETS.get(self.emotion.preset)
        strength = clamp(
            self.emotion.preset_intensity * self._gain("expression_gain", 1.0),
            0.0,
            1.0,
        )
        if expression and strength > 0.001:
            for key, value in expression.items():
                targets[key] = targets.get(key, 0.0) * (1.0 - strength) + value * strength
        return targets

    # -- head -------------------------------------------------------------

    def _head_targets(self, now: float) -> dict[str, float]:
        t = now - self._started
        gain = clamp(self._gain("head_gain", self.avatar.head_gain), 0.5, 3.5)
        # Idle liveliness: 有存在感 / 更活泼 一半一半 -> a bit more sway, still
        # bounded. It scales the idle component only, not emotion or speech.
        sway = 0.80 + 0.70 * self._idle_energy
        x = (
            1.80 * math.sin(2 * math.pi * 0.071 * t)
            + 0.60 * math.sin(2 * math.pi * 0.193 * t + 1.3)
        ) * sway
        y = 1.05 * math.sin(2 * math.pi * 0.113 * t + 0.7) * sway
        z = (
            2.00 * math.sin(2 * math.pi * 0.053 * t + 2.0)
            + 0.50 * math.sin(2 * math.pi * 0.227 * t + 0.4)
        ) * sway
        v = self.emotion.valence
        a = self.emotion.arousal
        # A small emotional bias; never a big pose change.
        y += (-2.8 * max(0.0, -v) + 1.0 * max(0.0, v)) * self._speech_blend
        z += (3.2 * max(0.0, -v) - 1.0 * max(0.0, v)) * self._speech_blend
        x += self._state_bias["x"] * self._speech_blend
        y += self._state_bias["y"] * self._speech_blend
        z += self._state_bias["z"] * self._speech_blend
        if self.use_affect and self.affect is not None:
            composed = compose_head(self.affect, self._gain("expression_head_gain", 1.0))
            if self.baseline is not None and self.baseline_weight > 0:
                base = compose_head(self.baseline, self._gain("expression_head_gain", 1.0))
                weight = self.baseline_weight
                for key in composed:
                    composed[key] = composed[key] * (1.0 - weight) + base.get(key, 0.0) * weight
            x += composed["yaw"]
            y += composed["pitch"]
            z += composed["roll"]
        elif self.use_affect and self.baseline is not None and self.baseline_weight > 0:
            composed = compose_head(
                self.baseline,
                self._gain("expression_head_gain", 1.0) * self.baseline_weight,
            )
            x += composed["yaw"]
            y += composed["pitch"]
            z += composed["roll"]
        else:
            expression_head = EXPRESSION_HEAD.get(self.emotion.preset, {})
            strength = clamp(
                self.emotion.preset_intensity * self._gain("expression_head_gain", 1.0),
                0.0,
                1.0,
            )
            x += expression_head.get("yaw", 0.0) * strength
            y += expression_head.get("pitch", 0.0) * strength
            z += expression_head.get("roll", 0.0) * strength
        return {"x": x * gain, "y": y * gain, "z": z * gain}

    def _update_state_bias(self, dt: float) -> None:
        """Ease the thinking/listening pose instead of switching it instantly."""
        target = {"x": 0.0, "y": 0.0, "z": 0.0}
        if self.state == "thinking":
            target = {"x": 1.4, "y": -2.0, "z": 5.0}
        elif self.state == "listening":
            target = {"x": 0.5, "y": 0.8, "z": 1.6}
        for key in ("x", "y", "z"):
            self._state_bias[key] = _smooth(
                self._state_bias[key], target[key], dt, 0.28, 0.55
            )

    def _action_targets(self, sample: int) -> dict[str, float]:
        out: dict[str, float] = {}
        active: list[ActionCue] = []
        for cue in self.actions:
            if sample >= cue.start + cue.duration:
                cue.finished = True
                continue
            if sample < cue.start:
                active.append(cue)
                continue
            values = cue.sample(sample)
            for key, value in values.items():
                out[key] = out.get(key, 0.0) + value
            active.append(cue)
        self.actions = active
        return out

    # -- smoothing --------------------------------------------------------

    def _sv(self, name: str, target: float, dt: float, attack: float, release: float) -> float:
        current = self._values.get(name, target)
        value = _smooth(current, target, dt, attack, release)
        self._values[name] = value
        return value

    def _spring(self, name: str, target: float, dt: float, *, freq: float = 1.25, zeta: float = 0.9) -> float:
        """Critically damped-ish spring for head/body motion.

        VTS head parameters are visible immediately; an EMA on a large nod
        target looks like a sudden snap. The spring carries velocity through
        the motion and settles without overshooting much.
        """
        current = self._values.get(name, target)
        velocity = self._vel.get(name, 0.0)
        omega = 2.0 * math.pi * freq
        accel = omega * omega * (target - current) - 2.0 * zeta * omega * velocity
        velocity += accel * dt
        current += velocity * dt
        self._values[name] = current
        self._vel[name] = velocity
        return current

    def _head_value(self, name: str, target: float, dt: float, smooth) -> float:
        """Rate-limit the head before any easing.

        This is the safety net: no state change, action cue, or model event can
        ask the neck to move faster than ``head_max_speed`` degrees/second.
        """
        current = self._values.get(name, target)
        if self.avatar.head_max_speed <= 0:
            return smooth(name, target)
        max_step = float(self.avatar.head_max_speed) * dt
        limited = current + clamp(target - current, -max_step, max_step)
        return smooth(name, limited)


def _silence() -> VisemeSegment:
    return VisemeSegment(0, 1, "_", "", "silence", 0.0, 0.0, 0.0)


def _energy(rms: float) -> float:
    db = 20.0 * math.log10(max(1e-6, float(rms)))
    floor_db = -45.0
    ceil_db = -12.0
    return clamp((db - floor_db) / (ceil_db - floor_db), 0.0, 1.0) ** 0.7
