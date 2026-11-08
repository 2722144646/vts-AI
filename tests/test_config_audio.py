from dsh_vts_avatar.config import _parse_audio_device


def test_default_keywords_mean_system_default():
    assert _parse_audio_device(None) is None
    assert _parse_audio_device("default") is None
    assert _parse_audio_device("DEFAULT") is None
    assert _parse_audio_device(" auto ") is None
    assert _parse_audio_device("") is None


def test_numeric_device_is_kept():
    assert _parse_audio_device(19) == 19
    assert _parse_audio_device("19") == 19
