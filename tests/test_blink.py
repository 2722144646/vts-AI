import time

from dsh_vts_avatar.config import AvatarConfig
from dsh_vts_avatar.performance.mixer import PerformanceMixer


def test_blink_fully_closes_eyes():
    mixer = PerformanceMixer(AvatarConfig(mouth_backend="continuous"), 48000)
    now = time.perf_counter()
    mixer.compute(0, None, now)
    eye_name = mixer.avatar.eye_left
    values = []
    for step in range(1, 15):
        t = now + step / 30.0
        # Pin the blink at its fully closed phase on every tick.
        mixer._blink._started = t - 0.42 * 0.16
        mixer._blink._next = t + 100.0
        params = mixer.compute(int(48000 * step / 30), None, t)
        values.append(params.get(eye_name, 1.0))
    assert min(values) < 0.05, values
