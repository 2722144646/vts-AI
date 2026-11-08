from pathlib import Path

from dsh_vts_avatar.bridge.engine import AvatarEngine
from dsh_vts_avatar.config import Config, EmotionConfig


class _StubMixer:
    def __init__(self) -> None:
        self.affect = None
        self.emotions: list[tuple[str, float]] = []
        self.actions: list = []

    def set_affect(self, state) -> None:
        self.affect = state

    def set_emotion(self, name: str, intensity: float = 0.7) -> None:
        self.emotions.append((name, intensity))

    def add_action(self, kind, intensity=0.7, **kwargs) -> None:
        self.actions.append((kind, intensity))


def _engine(mode: str, tmp_path: Path) -> AvatarEngine:
    config = Config(root=tmp_path)
    config.emotion = EmotionConfig(mode=mode)
    engine = AvatarEngine(config)
    engine.mixer = _StubMixer()
    return engine


def test_feel_blends_weighted_labels_in_affect_mode(tmp_path):
    engine = _engine("affect", tmp_path)
    result = engine.feel({"happy": 0.7, "tired": 0.3}, intensity=0.8, confidence=0.9)

    assert result["ok"] is True
    assert result["mode"] == "affect"
    assert abs(result["labels"]["happy"] - 0.7) < 1e-9
    assert result["dominant"][0][0] == "happy"
    assert result["valence"] > 0
    assert engine.mixer.affect is not None
    assert abs(engine.mixer.affect.labels["happy"] - 0.7) < 1e-9


def test_low_confidence_vote_fades_out_instead_of_hard_switching(tmp_path):
    engine = _engine("affect", tmp_path)
    # confidence 0 means the vote has no weight: it must not move the face.
    weak = engine.feel({"sad": 1.0}, intensity=1.0, confidence=0.0)
    assert weak["labels"] == {"neutral": 1.0}
    assert weak["intensity"] == 0.0

    confident = engine.feel({"sad": 1.0}, intensity=1.0, confidence=1.0)
    assert confident["labels"]["sad"] > 0.9
    assert confident["intensity"] > 0.9


def test_feel_degrades_to_dominant_preset_in_legacy_mode(tmp_path):
    engine = _engine("legacy", tmp_path)
    result = engine.feel({"sad": 0.9, "tired": 0.1}, intensity=0.7, confidence=0.8)

    assert result["mode"] == "legacy"
    assert engine.mixer.affect is None
    assert engine.mixer.emotions[-1][0] == "sad"


def test_feel_writes_a_replayable_affect_log(tmp_path):
    engine = _engine("affect", tmp_path)
    engine.feel({"happy": 1.0}, intensity=0.7, confidence=0.9, note="unit test")

    files = list((tmp_path / "data" / "affect").glob("affect-*.jsonl"))
    assert files, "feel must leave an evidence record on disk"
    lines = files[0].read_text(encoding="utf-8").splitlines()
    assert any('"note":"unit test"' in line for line in lines)
