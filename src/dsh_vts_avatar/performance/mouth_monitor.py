"""Parameter-side mouth health monitor.

The NDI ROI is useful for gross motion, but head sway contaminates it. This
monitor answers a narrower question: while speech is audible, are the mouth
parameters actually changing?
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


_MOUTH_KEYS = ("MouthOpen", "VoiceA", "VoiceI", "VoiceU", "VoiceE", "VoiceO")


@dataclass
class MouthMonitor:
    window_s: float = 1.0
    min_range: float = 0.01
    samples: deque[tuple[float, float]] = field(default_factory=deque)
    stuck: bool = False
    activity: float = 0.0

    def observe(self, now: float, params: dict[str, float], *, speaking: bool, audible: bool) -> None:
        if not speaking or not audible:
            self.samples.clear()
            self.stuck = False
            self.activity = 0.0
            return
        values = [float(params.get(key, 0.0)) for key in _MOUTH_KEYS]
        value = max(values, default=0.0)
        self.activity = value
        self.samples.append((now, value))
        cutoff = now - max(0.2, self.window_s)
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()
        if len(self.samples) < 4:
            self.stuck = False
            return
        low = min(item[1] for item in self.samples)
        high = max(item[1] for item in self.samples)
        self.stuck = (high - low) < self.min_range

    def reset(self) -> None:
        self.samples.clear()
        self.stuck = False
        self.activity = 0.0
