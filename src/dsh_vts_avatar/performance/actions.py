"""Short, procedural action cues (nods, shakes, surprise)."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random


def _ease_in_out(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def _ease_out(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return 1.0 - (1.0 - x) ** 3


def _minjerk(x: float) -> float:
    """5th-order minimum-jerk curve: zero velocity/acceleration at both ends."""
    x = max(0.0, min(1.0, x))
    return 10.0 * x**3 - 15.0 * x**4 + 6.0 * x**5


def _bezier(x: float, *, x1: float = 0.42, y1: float = 0.0, x2: float = 0.58, y2: float = 1.0) -> float:
    """Cubic Bezier ease with numerically solved time parameter."""
    x = max(0.0, min(1.0, x))
    lo, hi = 0.0, 1.0
    for _ in range(14):
        mid = (lo + hi) * 0.5
        bx = 3.0 * (1.0 - mid) ** 2 * mid * x1 + 3.0 * (1.0 - mid) * mid**2 * x2 + mid**3
        if bx < x:
            lo = mid
        else:
            hi = mid
    t = (lo + hi) * 0.5
    return 3.0 * (1.0 - t) ** 2 * t * y1 + 3.0 * (1.0 - t) * t**2 * y2 + t**3


def _hold(t: float, attack: float = 0.18, hold: float = 0.62) -> float:
    """Minimum-jerk attack, hold, minimum-jerk release."""
    t = max(0.0, min(1.0, t))
    if t < attack:
        return _minjerk(t / attack)
    if t < hold:
        return 1.0
    return 1.0 - _minjerk((t - hold) / max(1e-6, 1.0 - hold))


@dataclass
class ActionCue:
    kind: str
    start: int
    duration: int
    intensity: float = 0.7
    seed: int = 0
    curve: str = "minjerk"
    finished: bool = False

    @property
    def channel(self) -> str:
        if self.kind in {
            "nod", "nod_strong", "demo_nod", "shake", "shake_head",
            "surprise", "startle", "tilt", "thinking_tilt",
            "turn_look_left", "turn_look_right", "shy_avoid", "think_side",
            "emphasis",
        }:
            return "head"
        return "face"

    def sample(self, sample: int) -> dict[str, float]:
        if sample < self.start or sample >= self.start + self.duration:
            return {}
        t = (sample - self.start) / max(1, self.duration)
        amp = max(0.0, min(1.4, self.intensity))
        if self.kind.startswith("idle_"):
            return self._idle_sample(t)
        if self.kind == "demo_nod":
            curve = _bezier if self.curve == "bezier" else _minjerk
            if t < 0.45:
                pulse = curve(t / 0.45)
            else:
                pulse = 1.0 - curve((t - 0.45) / 0.55)
            return {"pitch": 8.0 * self.intensity * pulse}
        if self.kind in ("nod", "nod_strong"):
            # Anticipation -> down -> hold -> slow return.
            if t < 0.12:
                pitch = -1.5 * _minjerk(t / 0.12)
            elif t < 0.32:
                pitch = -1.5 + 9.5 * _minjerk((t - 0.12) / 0.20)
            elif t < 0.44:
                pitch = 8.0
            else:
                pitch = 8.0 * (1.0 - _minjerk((t - 0.44) / 0.56))
            return {"pitch": pitch * amp}
        if self.kind in ("shake", "shake_head"):
            if t < 0.1:
                return {"yaw": 0.0}
            yaw = math.sin((t - 0.1) * math.pi * 5.0) * 8.0 * (1.0 - t)
            return {"yaw": yaw * amp}
        if self.kind in ("surprise", "startle"):
            if t < 0.18:
                pitch = -10.0 * _minjerk(t / 0.18)
            else:
                pitch = -10.0 * (1.0 - _minjerk((t - 0.18) / 0.82))
            return {"pitch": pitch * amp, "eye_wide": amp}
        if self.kind in ("tilt", "thinking_tilt"):
            tilt = _minjerk(min(1.0, t * 2.0)) * (1.0 - _minjerk(max(0.0, (t - 0.5) * 2.0))) * 4.0
            return {"roll": tilt * amp}
        if self.kind == "emphasis":
            pitch = math.sin(t * math.pi) * 2.5
            return {"pitch": pitch * amp}
        if self.kind in ("turn_look_left", "turn_look_right"):
            # Eyes lead, head follows with a slight roll; a little brow/smile
            # keeps it from looking like a servo.
            direction = -1.0 if self.kind.endswith("left") else 1.0
            head = _hold(t, 0.16, 0.64)
            eyes = _hold(min(1.0, t * 1.15), 0.10, 0.60)
            settle = _hold(t, 0.30, 0.74)
            return {
                "yaw": 11.0 * direction * head * amp,
                "roll": -2.0 * direction * settle * amp,
                "pitch": 1.2 * settle * amp,
                "gaze_x": 0.58 * direction * eyes * amp,
                "gaze_y": 0.08 * settle * amp,
                "brow_l": 0.10 * settle * amp,
                "brow_r": 0.10 * settle * amp,
                "smile": 0.08 * settle * amp,
            }
        if self.kind in ("shy_avoid", "shy"):
            # Look down and away, smaller than a full turn; inner brows soften.
            head = _hold(t, 0.20, 0.62)
            eyes = _hold(min(1.0, t * 1.2), 0.12, 0.58)
            return {
                "yaw": -5.0 * head * amp,
                "pitch": -4.0 * head * amp,
                "roll": 3.5 * head * amp,
                "gaze_x": -0.42 * eyes * amp,
                "gaze_y": -0.50 * eyes * amp,
                "inner_brow": 0.30 * head * amp,
                "smile": 0.10 * head * amp,
            }
        if self.kind in ("think_side", "thinking_side"):
            # Head tilt and side gaze, with one brow a little higher.
            head = _hold(t, 0.22, 0.66)
            eyes = _hold(min(1.0, t * 1.10), 0.14, 0.62)
            return {
                "yaw": 3.0 * head * amp,
                "roll": 5.0 * head * amp,
                "pitch": 1.0 * head * amp,
                "gaze_x": 0.48 * eyes * amp,
                "gaze_y": 0.36 * eyes * amp,
                "brow_l": 0.16 * head * amp,
                "smile": 0.06 * head * amp,
            }
        if self.kind == "look_away":
            eyes = _hold(t, 0.12, 0.64)
            return {"gaze_x": 0.52 * eyes * amp, "gaze_y": 0.12 * eyes * amp}
        if self.kind == "sigh":
            # 哈气：张嘴呼气 + 低头 -> 慢慢回正。笑不归这里管。
            breathe = _hold(t, 0.08, 0.55)
            sink = _hold(t, 0.26, 0.70)
            return {
                "open": 0.50 * breathe * amp,
                "pitch": -7.0 * sink * amp,
            }
        return {}

    def _idle_sample(self, t: float) -> dict[str, float]:
        """Idle-library motions: small, slow, never a performance."""
        amp = max(0.0, min(1.4, self.intensity))
        hold = _hold(t, 0.12, 0.62)
        slow = _hold(t, 0.10, 0.72)
        if self.kind == "idle_glance":
            return {"gaze_x": 0.35 * hold * amp, "roll": 1.5 * hold * amp}
        if self.kind == "idle_look_down":
            return {"gaze_y": -0.45 * hold * amp, "pitch": 2.5 * hold * amp}
        if self.kind == "idle_look_away":
            return {
                "gaze_x": 0.42 * hold * amp,
                "gaze_y": 0.08 * hold * amp,
                "roll": 1.2 * hold * amp,
            }
        if self.kind == "idle_look_user":
            return {
                "gaze_y": 0.12 * hold * amp,
                "smile": 0.06 * hold * amp,
                "pitch": -0.8 * slow * amp,
            }
        if self.kind == "idle_tilt":
            return {"roll": 4.5 * hold * amp}
        if self.kind == "idle_nod":
            return {"pitch": 2.2 * hold * amp}
        if self.kind == "idle_breathe":
            return {"pitch": 1.0 * slow * amp, "gaze_y": 0.04 * slow * amp}
        if self.kind == "idle_sigh":
            return {"open": 0.25 * hold * amp, "pitch": 2.0 * hold * amp}
        if self.kind == "idle_hum":
            return {"pucker": 0.25 * hold * amp, "smile": 0.06 * hold * amp}
        if self.kind == "idle_yawn":
            return {"open": 0.45 * hold * amp, "pitch": 1.5 * hold * amp}
        return {}

    @classmethod
    def create(
        cls,
        kind: str,
        *,
        start: int,
        sample_rate: int,
        intensity: float = 0.7,
        duration_ms: int | None = None,
        curve: str = "minjerk",
    ) -> "ActionCue":
        durations = {
            "nod": 680,
            "nod_strong": 820,
            "shake": 700,
            "shake_head": 700,
            "surprise": 760,
            "startle": 760,
            "tilt": 900,
            "thinking_tilt": 1000,
            "turn_look_left": 1150,
            "turn_look_right": 1150,
            "shy_avoid": 1350,
            "think_side": 1300,
            "look_away": 950,
            "emphasis": 340,
            "sigh": 1800,
            "idle_glance": 1200,
            "idle_look_down": 1500,
            "idle_look_away": 1200,
            "idle_look_user": 1400,
            "idle_tilt": 1100,
            "idle_nod": 900,
            "idle_breathe": 1800,
            "idle_sigh": 1600,
            "idle_hum": 1500,
            "idle_yawn": 1800,
        }
        duration = duration_ms or durations.get(kind, 420)
        return cls(
            kind=kind,
            start=start,
            duration=max(1, int(sample_rate * duration / 1000)),
            intensity=max(0.0, min(1.4, intensity)),
            seed=random.randint(0, 1_000_000),
            curve=curve,
        )
