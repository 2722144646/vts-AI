import time

from dsh_vts_avatar.config import AvatarConfig
from dsh_vts_avatar.performance.emotion import EXPRESSION_TARGETS, EmotionState, emotion_from_text
from dsh_vts_avatar.performance.mixer import PerformanceMixer


def test_expression_presets_are_distinct():
    sad = EXPRESSION_TARGETS["sad"]
    happy = EXPRESSION_TARGETS["happy"]
    surprised = EXPRESSION_TARGETS["surprised"]
    assert sad["inner_brow"] > 0.8
    assert happy["smile"] > 0.8
    assert surprised["eye"] >= 1.0
    assert surprised["open"] > happy["open"]


def test_emotion_preset_blends_into_mixer_targets():
    mixer = PerformanceMixer(AvatarConfig(mouth_backend="continuous"), 48000)
    mixer.emotion.preset = "sad"
    mixer.emotion.preset_intensity = 0.9
    sad = mixer._emotion_targets()
    mixer.emotion.preset = "surprised"
    surprised = mixer._emotion_targets()
    assert sad["inner_brow"] > surprised["inner_brow"]
    assert surprised["eye"] > sad["eye"]
    assert surprised["open"] > sad["open"]


def test_thinking_keeps_improvised_preset_without_standard_take():
    from dsh_vts_avatar.vts.profile import ModelProfile

    profile = ModelProfile(model_id="t", expression_map={"思考": {"MouthSmile": 0.1}})
    mixer = PerformanceMixer(AvatarConfig(mouth_backend="continuous"), 48000, profile=profile)
    mixer.set_emotion("thinking", 1.0)
    assert mixer.emotion.preset == "thinking"


def test_learned_expression_alias_is_used():
    from dsh_vts_avatar.vts.profile import ModelProfile

    profile = ModelProfile(
        model_id="t",
        expression_map={"开心": {"MouthSmile": 1.0, "EyeOpenLeft": 1.0, "EyeOpenRight": 1.0}},
    )
    mixer = PerformanceMixer(AvatarConfig(mouth_backend="continuous"), 48000, profile=profile)
    mixer.set_emotion("happy", 1.0)
    assert mixer.emotion.preset == "开心"


def test_expression_gain_is_read_from_model_profile():
    from dsh_vts_avatar.vts.profile import ModelProfile

    profile = ModelProfile(model_id="t", tuning={"expression_gain": 0.5, "expression_head_gain": 0.5})
    mixer = PerformanceMixer(AvatarConfig(mouth_backend="continuous"), 48000, profile=profile)
    mixer.emotion.preset = "happy"
    mixer.emotion.preset_intensity = 1.0
    targets = mixer._emotion_targets()
    assert 0.5 < targets["smile"] < 0.9


def test_negation_and_intensifier_are_automatic():
    assert emotion_from_text("我不开心")[0] == "sad"
    assert emotion_from_text("不生气")[0] == "calm"
    assert emotion_from_text("太开心了")[1] > 0.8
    assert emotion_from_text("有点难过")[1] < 0.8


def test_direct_model_mapping_overrides_template():
    from dsh_vts_avatar.tts.timeline import Timeline
    from dsh_vts_avatar.vts.profile import ModelProfile

    profile = ModelProfile(
        model_id="t",
        tuning={"expression_gain": 1.0},
        expression_map={"happy": {"MouthSmile": 1.0, "EyeOpenLeft": 0.5, "EyeOpenRight": 0.5}},
    )
    mixer = PerformanceMixer(AvatarConfig(mouth_backend="continuous"), 48000, profile=profile)
    mixer.set_emotion("happy", 1.0)
    mixer.emotion.preset_intensity = 1.0
    timeline = Timeline.synthetic("你好", sample_rate=48000)
    now = time.perf_counter()
    for i in range(30):
        params = mixer.compute(1000 + i * 1600, timeline, now + i * 0.05)
    assert params["MouthSmile"] > 0.8
    assert params["EyeOpenLeft"] < 0.9
