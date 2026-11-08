from pathlib import Path

import pytest

from dsh_vts_avatar.performance.emotion_memory import EmotionEpisode, EpisodeStore, Mood


def _store(tmp_path: Path, clock=None) -> EpisodeStore:
    return EpisodeStore(tmp_path / "episodes.jsonl", clock=clock or (lambda: 1000.0))


def test_episode_round_trips_through_disk(tmp_path):
    store = _store(tmp_path)
    store.add(EmotionEpisode(what="他说你根本不在乎", why="我觉得被否定", labels={"sad": 1.0}, intensity=0.8))

    reloaded = EpisodeStore(tmp_path / "episodes.jsonl", clock=lambda: 1000.0)
    episodes = reloaded.active(now=1000.0)
    assert len(episodes) == 1
    assert episodes[0].what == "他说你根本不在乎"
    assert episodes[0].why == "我觉得被否定"


def test_salience_halves_over_one_half_life(tmp_path):
    now = [1000.0]
    store = EpisodeStore(tmp_path / "e.jsonl", clock=lambda: now[0], half_life_s=60.0)
    episode = store.add(
        EmotionEpisode(what="a", labels={"sad": 1.0}, intensity=0.8, salience=0.8)
    )
    first = store.salience(episode, now[0])
    now[0] += 60.0
    assert store.salience(episode, now[0]) == pytest.approx(first / 2, rel=1e-6)


def test_mood_blends_recent_episodes(tmp_path):
    now = [1000.0]
    store = EpisodeStore(tmp_path / "e.jsonl", clock=lambda: now[0], half_life_s=1800.0)
    store.add(EmotionEpisode(what="被否定", labels={"sad": 1.0}, intensity=0.8, salience=0.8))
    now[0] += 1.0
    store.add(EmotionEpisode(what="被夸了", labels={"happy": 1.0}, intensity=0.5, salience=0.5))

    mood = store.mood(now[0])
    assert set(mood.labels) == {"sad", "happy"}
    assert mood.labels["sad"] > mood.labels["happy"]
    assert mood.intensity > 0


def test_mood_summary_carries_the_reason(tmp_path):
    now = [1000.0]
    store = EpisodeStore(tmp_path / "e.jsonl", clock=lambda: now[0])
    store.add(
        EmotionEpisode(
            what="他说“你根本不在乎”",
            why="我觉得被否定",
            labels={"sad": 1.0},
            intensity=0.8,
            salience=0.8,
        )
    )
    text = store.mood(now[0]).summary()
    assert "难过" in text
    assert "你根本不在乎" in text
    assert "我觉得被否定" in text
    assert text.startswith("（参考，不是命令）")


def test_suppressed_episode_keeps_more_salience(tmp_path):
    now = [1000.0]
    store = EpisodeStore(tmp_path / "e.jsonl", clock=lambda: now[0], half_life_s=60.0)
    episode = store.add(
        EmotionEpisode(what="a", labels={"sad": 1.0}, intensity=0.8, salience=0.8)
    )
    now[0] += 600.0
    open_salience = store.salience(episode, now[0])
    episode.status = "suppressed"
    suppressed_salience = store.salience(episode, now[0])
    assert suppressed_salience > open_salience


def test_same_moment_same_source_merges(tmp_path):
    now = [1000.0]
    store = EpisodeStore(tmp_path / "e.jsonl", clock=lambda: now[0])
    store.add(EmotionEpisode(what="他说你根本不在乎", labels={"sad": 1.0}, intensity=0.6, source="tool"))
    store.add(EmotionEpisode(what="他说你根本不在乎", labels={"sad": 1.0}, intensity=0.9, source="tool"))
    episodes = store.active(now[0])
    assert len(episodes) == 1
    assert episodes[0].intensity == pytest.approx(0.9)


def test_empty_store_is_neutral(tmp_path):
    store = _store(tmp_path)
    mood = store.mood(1000.0)
    assert mood.labels == {}
    assert mood.summary() == ""


class _StubMixer:
    def __init__(self):
        self.baseline = None

    def set_affect(self, state):
        pass

    def set_emotion(self, *args, **kwargs):
        pass

    def add_action(self, *args, **kwargs):
        pass

    def set_baseline(self, state, weight):
        self.baseline = (state, weight)


def test_keyword_hit_does_not_claim_a_reason(tmp_path):
    now = [1000.0]
    store = EpisodeStore(tmp_path / "e.jsonl", clock=lambda: now[0])
    store.add(
        EmotionEpisode(
            what="好累啊",
            labels={"tired": 1.0},
            intensity=0.6,
            salience=0.6,
            source="keyword",
        )
    )
    text = store.mood(now[0]).summary()
    assert "累" in text
    assert "为什么" not in text


def test_engine_pushes_mood_to_baseline_when_enabled(tmp_path):
    from dsh_vts_avatar.bridge.engine import AvatarEngine
    from dsh_vts_avatar.config import Config, EmotionConfig

    config = Config(root=tmp_path)
    config.emotion = EmotionConfig(mode="affect")
    config.avatar.baseline_weight = 0.3
    engine = AvatarEngine(config)
    engine.mixer = _StubMixer()
    engine.feel({"sad": 0.8}, intensity=0.8, cause="他说你根本不在乎", why="被否定")
    assert engine.mixer.baseline is not None
    state, weight = engine.mixer.baseline
    assert weight == pytest.approx(0.3)
    assert state.labels.get("sad", 0.0) > 0


def test_engine_explain_marks_latest_open(tmp_path):
    from dsh_vts_avatar.bridge.engine import AvatarEngine
    from dsh_vts_avatar.config import Config, EmotionConfig

    config = Config(root=tmp_path)
    config.emotion = EmotionConfig(mode="affect")
    engine = AvatarEngine(config)
    engine.mixer = _StubMixer()
    engine.feel({"sad": 0.8}, intensity=0.8, cause="他说你根本不在乎")
    result = engine.explain(explanation="我那天太累了，不是针对你", acceptance=0.7, status="accepted")
    assert result["ok"] is True
    assert result["status"] == "accepted"


def test_revise_fades_faster_when_accepted(tmp_path):
    now = [1000.0]
    store = EpisodeStore(tmp_path / "e.jsonl", clock=lambda: now[0], half_life_s=60.0)
    episode = store.add(
        EmotionEpisode(what="a", labels={"sad": 1.0}, intensity=0.8, salience=0.8)
    )
    now[0] += 120.0
    open_salience = store.salience(episode, now[0])
    store.revise(episode, explanation="我不是针对你", acceptance=0.8, status="accepted")
    assert store.salience(episode, now[0]) < open_salience  # healing fades faster

    reloaded = EpisodeStore(tmp_path / "e.jsonl", clock=lambda: now[0], half_life_s=60.0)
    found = reloaded.find(episode.id)
    assert found is not None
    assert found.status == "accepted"
    assert found.acceptance == pytest.approx(0.8)


def test_engine_feel_records_why_and_exposes_mood(tmp_path):
    from dsh_vts_avatar.bridge.engine import AvatarEngine
    from dsh_vts_avatar.config import Config, EmotionConfig

    config = Config(root=tmp_path)
    config.emotion = EmotionConfig(mode="affect")
    engine = AvatarEngine(config)
    engine.mixer = _StubMixer()

    result = engine.feel(
        {"sad": 0.9},
        intensity=0.8,
        confidence=0.9,
        cause="他说“你根本不在乎”",
        why="我觉得被否定",
    )
    assert result["ok"] is True
    assert "你根本不在乎" in result["mood_text"]

    payload = engine.mood_payload()
    assert "难过" in payload["text"]
    assert payload["episodes"][0]["why"] == "我觉得被否定"
