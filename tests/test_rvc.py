import numpy as np

from dsh_vts_avatar.config import RVCConfig
from dsh_vts_avatar.tts.rvc import RVCConverter


def test_rvc_converter_is_noop_when_disabled():
    converter = RVCConverter(RVCConfig(enabled=False))
    assert converter.enabled is False


def test_rvc_resample_and_fit_keep_exact_timeline_length():
    samples = np.linspace(-0.5, 0.5, 40000, dtype=np.float32)
    resampled = RVCConverter._resample(samples, 40000, 22050)
    fitted = RVCConverter._fit(resampled, 36864)
    assert fitted.size == 36864
    assert fitted.dtype == np.float32
