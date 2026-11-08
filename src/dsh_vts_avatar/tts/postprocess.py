"""Light post-processing for Piper speech.

The selected A/B preset: high-pass, gentle low-mid cut, presence lift and a
2.5:1 compressor, followed by peak normalization. Duration is preserved.
"""

from __future__ import annotations

import logging
from pathlib import Path
import subprocess
import tempfile
import wave

import numpy as np

from ..config import TTSConfig
from .timeline import Timeline

FILTERS = (
    "highpass=f=75,"
    "equalizer=f=300:t=q:w=1.0:g=-1.5,"
    "equalizer=f=3200:t=q:w=1.2:g=2.5,"
    "acompressor=threshold=-18dB:ratio=2.5:attack=10:release=120:makeup=2"
)


class AudioPostProcessor:
    def __init__(self, config: TTSConfig, *, logger: logging.Logger | None = None):
        self.config = config
        self.log = logger or logging.getLogger(__name__)

    def _read(self, path: Path) -> tuple[np.ndarray, int]:
        with wave.open(str(path), "rb") as fh:
            rate = fh.getframerate()
            channels = fh.getnchannels()
            samples = np.frombuffer(
                fh.readframes(fh.getnframes()), dtype="<i2"
            ).astype(np.float32) / 32768.0
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)
        return samples, rate

    @staticmethod
    def _write(path: Path, samples: np.ndarray, rate: int) -> None:
        with wave.open(str(path), "wb") as fh:
            fh.setnchannels(1)
            fh.setsampwidth(2)
            fh.setframerate(rate)
            fh.writeframes((np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes())

    @staticmethod
    def _normalize(samples: np.ndarray, target_db: float = -1.0) -> np.ndarray:
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        if peak <= 1e-9:
            return samples
        return (samples * (10 ** (target_db / 20.0) / peak)).astype(np.float32)

    @staticmethod
    def _fit(samples: np.ndarray, size: int) -> np.ndarray:
        if samples.size >= size:
            return samples[:size].astype(np.float32, copy=False)
        return np.pad(samples, (0, size - samples.size)).astype(np.float32, copy=False)

    def process(self, timeline: Timeline) -> Timeline:
        if not self.config.postprocess:
            return timeline
        ffmpeg = Path(self.config.ffmpeg_path)
        if not ffmpeg.is_file():
            self.log.warning("postprocess ffmpeg not found: %s", ffmpeg)
            return timeline
        try:
            with tempfile.TemporaryDirectory(prefix="dsh-tts-") as tmp:
                tmp_path = Path(tmp)
                source = tmp_path / "in.wav"
                target = tmp_path / "out.wav"
                self._write(source, timeline.audio, timeline.sample_rate)
                subprocess.run(
                    [
                        str(ffmpeg), "-y", "-i", str(source),
                        "-af", FILTERS, "-ar", str(timeline.sample_rate), "-ac", "1",
                        str(target),
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=20,
                )
                samples, rate = self._read(target)
                samples = self._fit(
                    self._resample(samples, rate, timeline.sample_rate),
                    timeline.audio.size,
                )
                samples = self._normalize(samples)
                return self._replace_audio(timeline, samples)
        except Exception as exc:
            self.log.warning("postprocess failed; using raw Piper audio: %s", exc)
            return timeline

    @staticmethod
    def _resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
        if source_rate == target_rate or samples.size == 0:
            return samples.astype(np.float32, copy=False)
        target_size = max(1, int(round(samples.size * target_rate / source_rate)))
        old_t = np.arange(samples.size, dtype=np.float64) / source_rate
        new_t = np.arange(target_size, dtype=np.float64) / target_rate
        return np.interp(new_t, old_t, samples).astype(np.float32)

    def _replace_audio(self, timeline: Timeline, samples: np.ndarray) -> Timeline:
        hop = max(1, int(timeline.sample_rate * 0.01))
        count = max(1, samples.size // hop)
        frames = samples[: count * hop].reshape(count, hop)
        rms = np.sqrt(np.mean(frames * frames, axis=1)).astype(np.float32)
        return Timeline(
            sample_rate=timeline.sample_rate,
            audio=samples.astype(np.float32, copy=False),
            spans=timeline.spans,
            segments=timeline.segments,
            rms=rms,
            rms_hop=hop,
            language=timeline.language,
        )
