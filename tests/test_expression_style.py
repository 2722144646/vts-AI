from pathlib import Path

from dsh_vts_avatar.bridge.engine import AvatarEngine
from dsh_vts_avatar.config import AvatarConfig, Config
from dsh_vts_avatar.performance.mixer import GazeController, PerformanceMixer


class _FixedRandom:
    def __init__(self, value: float) -> None:
        self.value = value

    def random(self) -> float:
        return self.value


def _engine(tmp_path: Path) -> AvatarEngine:
    config = Config(root=tmp_path)
    engine = AvatarEngine(config)
    return engine


def test_default_style_matches_the_user_answer():
    avatar = AvatarConfig()
    # 有存在感 / 更活泼 一半一半；自然 70% / 夸张 30%
    assert 0.4 <= avatar.idle_energy <= 0.7
    assert abs(avatar.exaggeration_share - 0.30) < 1e-9
    assert avatar.exaggeration_gain > 1.0


def test_expression_gain_samples_natural_and_exaggerated(tmp_path):
    engine = _engine(tmp_path)
    engine._rng = _FixedRandom(0.0)  # inside the 30% share
    assert engine._sample_expression_gain() > 1.0

    engine._rng = _FixedRandom(0.99)  # natural 70%
    assert engine._sample_expression_gain() == 1.0


def test_idle_energy_scales_gaze_and_head_sway():
    calm = AvatarConfig(idle_energy=0.0, mouth_backend="continuous")
    lively = AvatarConfig(idle_energy=1.0, mouth_backend="continuous")

    calm_mixer = PerformanceMixer(calm, 48000)
    lively_mixer = PerformanceMixer(lively, 48000)
    calm_mixer._started = lively_mixer._started = 0.0

    calm_head = calm_mixer._head_targets(3.0)
    lively_head = lively_mixer._head_targets(3.0)
    assert abs(lively_head["x"]) > abs(calm_head["x"])

    assert isinstance(calm_mixer._gaze, GazeController)
    assert lively_mixer._gaze._amp > calm_mixer._gaze._amp
