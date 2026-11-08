from dsh_vts_avatar.config import AvatarConfig
from dsh_vts_avatar.performance.mixer import PerformanceMixer
from dsh_vts_avatar.tts.timeline import VisemeSegment
from dsh_vts_avatar.tts.visemes import map_phoneme


def _segment(phoneme: str, start: int = 0, duration: int = 4800) -> VisemeSegment:
    spec = map_phoneme(phoneme, "zh")
    return VisemeSegment(
        start,
        start + duration,
        phoneme,
        spec.viseme,
        spec.kind,
        spec.open,
        spec.form,
        spec.vowel_gain,
    )


def test_diphthong_travels_from_open_to_closed():
    mixer = PerformanceMixer(AvatarConfig(mouth_backend="continuous"), 48000)
    seg = _segment("ai")
    early = mixer._speech(seg, 0.6, 0)
    late = mixer._speech(seg, 0.6, seg.end - 1)
    assert late["open"] < early["open"]
    assert late["form"] != early["form"]


def test_plain_vowel_is_static():
    mixer = PerformanceMixer(AvatarConfig(mouth_backend="continuous"), 48000)
    seg = _segment("a")
    early = mixer._speech(seg, 0.6, 0)
    late = mixer._speech(seg, 0.6, seg.end - 1)
    assert early == late
