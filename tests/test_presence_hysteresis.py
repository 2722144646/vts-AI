from dsh_vts_avatar.performance.presence import PresenceDirector


def test_bucket_has_hysteresis_at_bright_threshold():
    director = PresenceDirector()
    director.observe({"mood_labels": {"happy": 0.46}, "mood_intensity": 0.8})
    assert director.profile.key == "bright"
    # dip just below the entry threshold: stays bright
    director.observe({"mood_labels": {"happy": 0.43}, "mood_intensity": 0.8})
    assert director.profile.key == "bright"
    # below the exit threshold: leaves
    director.observe({"mood_labels": {"happy": 0.30}, "mood_intensity": 0.8})
    assert director.profile.key == "calm"
