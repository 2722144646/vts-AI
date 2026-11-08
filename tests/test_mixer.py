import time

from dsh_vts_avatar.config import AvatarConfig
from dsh_vts_avatar.performance.mixer import BlinkController, PerformanceMixer
from dsh_vts_avatar.tts.timeline import Timeline


def test_mixer_returns_core_params_and_ranges():
    avatar = AvatarConfig(mouth_backend="continuous")
    timeline = Timeline.synthetic("你好，世界。", sample_rate=48000)
    mixer = PerformanceMixer(avatar, 48000)
    params = mixer.compute(1000, timeline, time.perf_counter())
    for key in ("MouthOpen", "MouthSmile", "EyeOpenLeft", "EyeOpenRight", "FaceAngleX"):
        assert key in params
        assert -30.0 <= params[key] <= 30.0
    assert 0.0 <= params["MouthOpen"] <= 1.0
    assert -1.0 <= params["MouthSmile"] <= 1.0


def test_blink_returns_to_open():
    blink = BlinkController(per_minute=60, seed=1)
    now = time.perf_counter()
    # Force a blink by moving time past the next scheduled blink.
    blink._next = now
    closed = blink.update(now + 0.02)
    assert 0.0 <= closed <= 1.0
    opened = blink.update(now + 1.0)
    assert opened == 1.0


def test_action_does_not_break_ranges():
    avatar = AvatarConfig(mouth_backend="continuous", head_gain=1.0)
    mixer = PerformanceMixer(avatar, 48000)
    mixer.add_action("nod", 1.0, at_sample=0)
    params = mixer.compute(10000, Timeline.synthetic("a", sample_rate=48000), time.perf_counter())
    assert -30.0 <= params["FaceAngleY"] <= 30.0
