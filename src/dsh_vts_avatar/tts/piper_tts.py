"""Piper TTS adapter with alignment validation."""

from __future__ import annotations

import logging
from pathlib import Path

from ..config import TTSConfig
from .postprocess import AudioPostProcessor
from .timeline import Timeline

log = logging.getLogger(__name__)


class PiperAlignmentError(RuntimeError):
    pass


class PiperTTS:
    def __init__(self, config: TTSConfig, *, logger: logging.Logger | None = None):
        self.config = config
        self.log = logger or log
        self.voice = None
        self._loaded = False
        self.sample_rate = 22050
        self.postprocessor = AudioPostProcessor(config, logger=self.log)

    @property
    def available(self) -> bool:
        return self.config.voice_model.exists() and (
            self.config.voice_config is None or self.config.voice_config.exists()
        )

    def load(self) -> None:
        if self._loaded:
            return
        if not self.config.voice_model.exists():
            raise FileNotFoundError(f"Piper model not found: {self.config.voice_model}")
        config_path = self.config.voice_config
        if config_path is not None and not config_path.exists():
            raise FileNotFoundError(f"Piper config not found: {config_path}")
        from piper import PiperVoice

        if config_path is not None:
            try:
                import json

                raw = json.loads(config_path.read_text(encoding="utf-8"))
                self.sample_rate = int(raw.get("audio", {}).get("sample_rate", self.sample_rate))
            except Exception:
                pass
        self.log.info("loading Piper voice %s (sr=%d)", self.config.voice_model, self.sample_rate)
        self.voice = PiperVoice.load(
            self.config.voice_model,
            config_path=config_path,
            include_alignments=True,
        )
        self._loaded = True
        self.log.info("Piper voice loaded")

    def synthesize(self, text: str) -> Timeline:
        self.load()
        assert self.voice is not None
        from piper.config import SynthesisConfig

        synth_kwargs = {}
        if self.config.noise_scale is not None:
            synth_kwargs["noise_scale"] = float(self.config.noise_scale)
        if self.config.noise_w_scale is not None:
            synth_kwargs["noise_w_scale"] = float(self.config.noise_w_scale)
        synth_config = SynthesisConfig(
            length_scale=max(0.30, min(3.0, float(self.config.length_scale))),
            normalize_audio=True,
            **synth_kwargs,
        )
        timelines: list[Timeline] = []
        for chunk in self.voice.synthesize(
            text,
            syn_config=synth_config,
            include_alignments=True,
        ):
            alignments = getattr(chunk, "phoneme_alignments", None)
            if not alignments:
                raise PiperAlignmentError(
                    "Piper returned no phoneme alignments. Patch the ONNX voice or load it with "
                    "include_alignments=True (and install piper-tts[alignment])."
                )
            audio = getattr(chunk, "audio_float_array", None)
            if audio is None:
                raise PiperAlignmentError("Piper chunk has no audio_float_array")
            sample_rate = int(getattr(chunk, "sample_rate", 22050))
            timeline = Timeline.from_alignments(
                audio,
                sample_rate,
                alignments,
                language=self.config.language,
            )
            timelines.append(timeline)
        if not timelines:
            raise PiperAlignmentError("Piper produced no audio chunks")
        if len(timelines) == 1:
            timeline = timelines[0]
        else:
            timeline = Timeline.concat(timelines)
        return self.postprocessor.process(timeline)

    def close(self) -> None:
        self.voice = None
        self._loaded = False
