from dsh_vts_avatar.audio.engine import AudioEngine


class _FakeSD:
    def query_devices(self, device=None, kind=None):
        if device is None:
            return [{"name": "default"}, {"name": "speaker"}]
        return {"name": "speaker", "max_output_channels": 2}


def test_device_name_handles_device_list_without_device():
    engine = object.__new__(AudioEngine)
    engine._sd = _FakeSD()
    assert engine._device_name(None) == ""
    assert engine._device_name(3) == "speaker"
