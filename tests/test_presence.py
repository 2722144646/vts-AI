import random

from dsh_vts_avatar.performance.actions import ActionCue
from dsh_vts_avatar.performance.presence import (
    MOUTH_KINDS,
    POOLS,
    PresenceDirector,
    idle_mouth_allowed,
)


def test_profile_buckets_follow_mood_and_relationship():
    director = PresenceDirector(base_energy=0.55)
    assert director.observe({
        "mood_labels": {"happy": 0.8}, "mood_intensity": 0.8,
        "trust": 0.6, "closeness": 0.6,
    }).key == "bright"
    assert director.observe({
        "mood_labels": {"sad": 0.7}, "mood_intensity": 0.7,
    }).key == "low"
    assert director.observe({
        "mood_labels": {}, "friction": 0.8,
    }).key == "tense"
    assert director.observe({
        "mood_labels": {"calm": 0.5}, "mood_intensity": 0.3,
        "trust": 0.9, "closeness": 0.8,
    }).key == "cozy"


def test_every_pool_kind_is_a_real_action():
    for pool in POOLS.values():
        for kind in pool:
            cue = ActionCue.create(kind, start=0, sample_rate=1000)
            assert cue.sample(300), kind


def test_idle_mouth_only_when_mouth_is_free():
    assert idle_mouth_allowed("idle", False)
    assert not idle_mouth_allowed("speaking", False)
    assert not idle_mouth_allowed("idle", True)


def test_director_does_not_repeat_the_last_two():
    director = PresenceDirector(rng=random.Random(7))
    director.observe({"mood_labels": {"happy": 0.8}, "mood_intensity": 0.8})
    picks = []
    for step in range(30):
        cue = director.next_cue(1000.0 + step * 20.0, speaking=False)
        if cue:
            picks.append(cue[0])
    assert picks
    for before, after in zip(picks, picks[1:]):
        assert before != after


def test_director_says_nothing_while_speaking():
    director = PresenceDirector(rng=random.Random(3))
    director.observe({"mood_labels": {"calm": 0.5}, "mood_intensity": 0.3})
    assert director.next_cue(1000.0, speaking=True) is None


def test_tense_pool_never_touches_the_mouth():
    director = PresenceDirector(rng=random.Random(5))
    director.observe({"mood_labels": {}, "friction": 0.9})
    picks = set()
    for step in range(30):
        cue = director.next_cue(2000.0 + step * 20.0, speaking=False)
        if cue:
            picks.add(cue[0])
    assert picks
    assert not (picks & MOUTH_KINDS)

def test_presence_batch_save_load_and_refresh_flags(tmp_path):
    from dsh_vts_avatar.performance.presence import PresenceDirector

    path = tmp_path / "idle_library.json"
    director = PresenceDirector(library_path=str(path))
    director.observe({"mood_labels": {"happy": 0.8}, "mood_intensity": 0.8})
    assert director.needs_write is True
    result = director.save_batch(
        "bright",
        [
            {"kind": "idle_tilt", "weight": 2.0, "intensity": 0.6},
            {"kind": "idle_hum", "weight": 1.0, "intensity": 0.5},
        ],
        note="happy day",
    )
    assert result["ok"] is True
    assert director.needs_write is False
    assert path.exists()

    reloaded = PresenceDirector(library_path=str(path))
    assert "bright" in reloaded.library
    assert reloaded.needs_write is False

    reloaded.observe({"mood_labels": {"sad": 0.8}, "mood_intensity": 0.8})
    assert reloaded.needs_write is True
    assert reloaded.write_reason == "profile_change"


def test_presence_batch_expires_after_24h(tmp_path):
    from dsh_vts_avatar.performance.presence import PresenceDirector

    director = PresenceDirector(library_path=str(tmp_path / "lib.json"))
    director.save_batch("calm", [{"kind": "idle_glance", "weight": 1.0, "intensity": 0.5}])
    assert director.needs_write is False
    assert director.check_expiry(director.generated_at + 86401) is True
    assert director.write_reason == "expired"


def test_presence_rejects_unknown_action(tmp_path):
    import pytest

    from dsh_vts_avatar.performance.presence import PresenceDirector

    director = PresenceDirector(library_path=str(tmp_path / "lib.json"))
    with pytest.raises(ValueError):
        director.save_batch("calm", [{"kind": "idle_nope", "weight": 1.0}])
