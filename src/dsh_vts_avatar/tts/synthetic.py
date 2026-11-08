"""Silent fallback TTS used only to test the VTS parameter path."""

from __future__ import annotations

from .timeline import Timeline


class SyntheticTTS:
    def __init__(self, *, language: str = "zh", sample_rate: int = 48000):
        self.language = language
        self.sample_rate = sample_rate

    @property
    def available(self) -> bool:
        return True

    def synthesize(self, text: str) -> Timeline:
        return Timeline.synthetic(
            text,
            sample_rate=self.sample_rate,
            language=self.language,
        )

    def close(self) -> None:
        return None
