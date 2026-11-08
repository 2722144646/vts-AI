import random
from dsh_vts_avatar.performance.presence import next_gap, quiet_bucket

def test_silence_slows_the_idle_cadence():
    rng1 = random.Random(1)
    rng2 = random.Random(1)
    normal = next_gap(0.0, rng1)
    quiet = next_gap(600.0, rng2)
    assert quiet > normal >= 6.0

def test_quiet_buckets():
    assert quiet_bucket(10) == "normal"
    assert quiet_bucket(120) == "quiet"
    assert quiet_bucket(400) == "very_quiet"
