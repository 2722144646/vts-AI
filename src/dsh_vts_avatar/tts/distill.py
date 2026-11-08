"""Distilled Piper voice with reference-timed audio.

Huayan supplies the target duration; the distilled voice supplies timbre. If
the candidate's duration differs, only its audio is time-stretched and its own
phoneme spans are scaled with it, so mouth timing stays internally consistent.
"""

from __future__ import annotations

from dataclasses import replace
import subprocess
import tempfile
from pathlib import Path
import wave

import numpy as np

from ..config import TTSConfig
from .piper_tts import PiperTTS
from .timeline import PhonemeSpan, Timeline, VisemeSegment


class DistilledPiperTTS:
    def __init__(self, config: TTSConfig, *, logger=None):
        self.config = config
        self.reference = PiperTTS(config, logger=logger)
        candidate_config = replace(
            config,
            voice_model=config.distill_model,
            voice_config=config.distill_voice_config,
            length_scale=config.distill_length_scale,
            postprocess=False,
        )
        self.candidate = PiperTTS(candidate_config, logger=logger)
        self.sample_rate = self.reference.sample_rate
        self.ffmpeg = Path(config.ffmpeg_path)
        self.last_factor = 1.0
        self.last_reference_duration = 0.0

    @property
    def available(self) -> bool:
        return self.reference.available and self.candidate.available

    def load(self) -> None:
        self.reference.load()
        self.candidate.load()
        self.sample_rate = self.reference.sample_rate

    def synthesize(self, text: str) -> Timeline:
        reference = self.reference.synthesize(text)
        candidate = self.candidate.synthesize(text)
        target_size = reference.audio.size
        factor = reference.duration_s / max(1e-6, candidate.duration_s)
        self.last_factor = factor
        self.last_reference_duration = reference.duration_s
        if abs(factor - 1.0) <= 0.02:
            return candidate
        speed = 1.0 / max(0.4, min(2.5, factor))
        audio = self._atempo(candidate.audio, candidate.sample_rate, speed)
        audio = self._fit(audio, target_size)
        return self._scale_timeline(candidate, factor, audio)

    def _atempo(self, audio: np.ndarray, rate: int, speed: float) -> np.ndarray:
        if abs(speed - 1.0) < 0.005:
            return audio.astype(np.float32, copy=False)
        with tempfile.TemporaryDirectory(prefix="dsh-distill-") as tmp:
            source = Path(tmp) / "in.wav"
            target = Path(tmp) / "out.wav"
            self._write(source, audio, rate)
            subprocess.run(
                [
                    str(self.ffmpeg), "-y", "-i", str(source),
                    "-filter:a", f"atempo={speed:.6f}",
                    "-ar", str(rate), "-ac", "1", str(target),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
            )
            result, _ = self._read(target)
        return result

    @staticmethod
    def _fit(audio: np.ndarray, size: int) -> np.ndarray:
        if audio.size >= size:
            return audio[:size].astype(np.float32, copy=False)
        return np.pad(audio, (0, size - audio.size)).astype(np.float32, copy=False)

    @staticmethod
    def _read(path: Path) -> tuple[np.ndarray, int]:
        with wave.open(str(path), "rb") as fh:
            rate = fh.getframerate()
            audio = np.frombuffer(fh.readframes(fh.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
        return audio, rate

    @staticmethod
    def _write(path: Path, audio: np.ndarray, rate: int) -> None:
        with wave.open(str(path), "wb") as fh:
            fh.setnchannels(1)
            fh.setsampwidth(2)
            fh.setframerate(rate)
            fh.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())

    @staticmethod
    def _scale_timeline(timeline: Timeline, factor: float, audio: np.ndarray) -> Timeline:
        spans = [
            PhonemeSpan(
                span.phoneme,
                int(round(span.start * factor)),
                min(audio.size, int(round(span.end * factor))),
            )
            for span in timeline.spans
        ]
        segments = [
            VisemeSegment(
                int(round(seg.start * factor)),
                min(audio.size, int(round(seg.end * factor))),
                seg.phoneme,
                seg.viseme,
                seg.kind,
                seg.open,
                seg.form,
                seg.vowel_gain,
            )
            for seg in timeline.segments
        ]
        hop = max(1, int(timeline.sample_rate * 0.01))
        count = max(1, audio.size // hop)
        frames = audio[: count * hop].reshape(count, hop)
        rms = np.sqrt(np.mean(frames * frames, axis=1)).astype(np.float32)
        return Timeline(
            sample_rate=timeline.sample_rate,
            audio=audio.astype(np.float32, copy=False),
            spans=spans,
            segments=segments,
            rms=rms,
            rms_hop=hop,
            language=timeline.language,
        )
