"""TTS adapters. Piper is the MVP engine because it exposes phoneme timings."""

from .piper_tts import PiperTTS, PiperAlignmentError
from .synthetic import SyntheticTTS

__all__ = ["PiperTTS", "PiperAlignmentError", "SyntheticTTS"]
