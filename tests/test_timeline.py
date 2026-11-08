from dsh_vts_avatar.tts.timeline import Timeline
from dsh_vts_avatar.tts.visemes import map_phoneme


def test_viseme_vowels_and_bilabial():
    assert map_phoneme("a").viseme == "A"
    assert map_phoneme("i").viseme == "I"
    assert map_phoneme("u").viseme == "U"
    assert map_phoneme("e").viseme == "E"
    assert map_phoneme("o").viseme == "O"
    assert map_phoneme("m").kind == "consonant"
    assert map_phoneme("m").open < 0.1
    assert map_phoneme("_").kind == "silence"


def test_synthetic_timeline_ranges():
    timeline = Timeline.synthetic("hello world", sample_rate=48000)
    assert timeline.duration_samples > 0
    first = timeline.at(0)
    assert first.start == 0
    assert first.kind in {"vowel", "consonant"}
    end = timeline.at(timeline.duration_samples - 1)
    assert end.end <= timeline.duration_samples


def test_concat_offsets():
    a = Timeline.synthetic("a", sample_rate=48000, phone_ms=100)
    b = Timeline.synthetic("i", sample_rate=48000, phone_ms=100)
    combined = Timeline.concat([a, b])
    assert combined.duration_samples == a.duration_samples + b.duration_samples
    assert combined.at(a.duration_samples).start >= a.duration_samples


def test_alignment_sum_mismatch_raises():
    class Align:
        def __init__(self, phoneme, num_samples):
            self.phoneme = phoneme
            self.num_samples = num_samples

    import numpy as np
    import pytest

    with pytest.raises(ValueError):
        Timeline.from_alignments(
            np.zeros(100, dtype=np.float32),
            48000,
            [Align("a", 50)],
        )
