from dsh_vts_avatar.performance.actions import ActionCue
from dsh_vts_avatar.config import AvatarConfig
from dsh_vts_avatar.performance.mixer import PerformanceMixer


def test_overlapping_head_actions_are_suppressed():
    mixer = PerformanceMixer(AvatarConfig(mouth_backend="continuous"), 48000)
    assert mixer.add_action("nod", 0.5, at_sample=0) is not None
    assert mixer.add_action("turn_look_left", 0.5, at_sample=1000) is None


def test_turn_look_has_cooldown():
    mixer = PerformanceMixer(AvatarConfig(mouth_backend="continuous"), 48000)
    assert mixer.add_action("turn_look_left", 0.5, at_sample=0) is not None
    assert mixer.add_action("turn_look_right", 0.5, at_sample=48000) is None
    assert mixer.add_action("turn_look_right", 0.5, at_sample=144000) is not None


def test_turn_look_combines_head_and_eyes():
    cue = ActionCue.create("turn_look_left", start=0, sample_rate=48000, intensity=1.0)
    values = cue.sample(24000)
    assert values["yaw"] < 0
    assert values["gaze_x"] < 0
    assert values["roll"] > 0 or values["roll"] < 0


def test_shy_and_think_actions_have_distinct_gaze():
    shy = ActionCue.create("shy_avoid", start=0, sample_rate=48000, intensity=1.0).sample(30000)
    think = ActionCue.create("think_side", start=0, sample_rate=48000, intensity=1.0).sample(30000)
    assert shy["gaze_y"] < 0
    assert think["gaze_y"] > 0
    assert shy["inner_brow"] > 0
