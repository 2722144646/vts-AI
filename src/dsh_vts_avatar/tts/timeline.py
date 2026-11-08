"""Audio + phoneme timeline used by the performance mixer."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
import math
from typing import Iterable, Sequence

import numpy as np

from .visemes import VisemeSpec, map_phoneme


@dataclass(frozen=True, slots=True)
class PhonemeSpan:
    phoneme: str
    start: int
    end: int

    @property
    def samples(self) -> int:
        return max(0, self.end - self.start)


@dataclass(frozen=True, slots=True)
class VisemeSegment:
    start: int
    end: int
    phoneme: str
    viseme: str
    kind: str
    open: float
    form: float
    vowel_gain: float

    @property
    def samples(self) -> int:
        return max(0, self.end - self.start)


_SILENCE_PHONES = {"^", "$", "_", " ", ".", ",", "!", "?", ";", ":", "-", "…"}
_TRANSPARENT_PHONES = {"ˈ", "ˌ", "ː", "ˑ", "1", "2", "3", "4", "5"}


@dataclass(slots=True)
class Timeline:
    sample_rate: int
    audio: np.ndarray
    spans: list[PhonemeSpan]
    segments: list[VisemeSegment]
    rms: np.ndarray
    rms_hop: int
    language: str = "zh"
    _starts: list[int] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self.audio = np.asarray(self.audio, dtype=np.float32).reshape(-1)
        self.rms = np.asarray(self.rms, dtype=np.float32).reshape(-1)
        self.segments.sort(key=lambda segment: segment.start)
        self._starts = [segment.start for segment in self.segments]

    @property
    def duration_samples(self) -> int:
        return int(self.audio.size)

    @property
    def duration_s(self) -> float:
        return self.duration_samples / max(1, self.sample_rate)

    def at(self, sample: float | int) -> VisemeSegment:
        if not self.segments:
            return _silence_segment(0, 1)
        s = int(max(0, min(sample, self.duration_samples - 1)))
        index = bisect_right(self._starts, s) - 1
        if index < 0:
            return _silence_segment(0, max(1, self.segments[0].start))
        segment = self.segments[index]
        if s >= segment.end:
            return _silence_segment(segment.end, segment.end + 1)
        return segment

    def bilabial_at(self, sample: float | int, window_samples: int = 960) -> VisemeSegment | None:
        """Return a nearby /p/ /b/ /m/ that a 30Hz control frame can show.

        Piper gives many bilabials only 10-30ms. A 33ms frame can land entirely
        between two of them; the mouth then never closes for the visible word.
        A small look-around window preserves the closure without changing the
        vowel timeline.
        """
        if not self.segments:
            return None
        s = int(max(0, min(sample, self.duration_samples - 1)))
        lo = s - max(0, window_samples)
        hi = s + max(0, window_samples)
        index = max(0, bisect_right(self._starts, lo) - 1)
        for segment in self.segments[index:]:
            if segment.start > hi:
                break
            if segment.end < lo:
                continue
            if (segment.phoneme or "").lower() in {"p", "b", "m"}:
                return segment
        return None

    def rms_at(self, sample: float | int) -> float:
        if self.rms.size == 0:
            return 0.0
        index = int(max(0, sample) / max(1, self.rms_hop))
        if index >= self.rms.size:
            return 0.0
        return float(self.rms[index])

    @classmethod
    def concat(cls, timelines: Sequence["Timeline"], *, gap_s: float = 0.0) -> "Timeline":
        if not timelines:
            raise ValueError("no timelines to concatenate")
        sr = timelines[0].sample_rate
        if any(t.sample_rate != sr for t in timelines):
            raise ValueError("cannot concatenate timelines with different sample rates")
        audio_parts: list[np.ndarray] = []
        spans: list[PhonemeSpan] = []
        segments: list[VisemeSegment] = []
        offset = 0
        for index, timeline in enumerate(timelines):
            if index and gap_s > 0:
                gap = np.zeros(int(sr * gap_s), dtype=np.float32)
                audio_parts.append(gap)
                offset += gap.size
            audio_parts.append(timeline.audio)
            for span in timeline.spans:
                spans.append(PhonemeSpan(span.phoneme, span.start + offset, span.end + offset))
            for segment in timeline.segments:
                segments.append(
                    VisemeSegment(
                        segment.start + offset,
                        segment.end + offset,
                        segment.phoneme,
                        segment.viseme,
                        segment.kind,
                        segment.open,
                        segment.form,
                        segment.vowel_gain,
                    )
                )
            offset += timeline.audio.size
        audio = np.concatenate(audio_parts) if audio_parts else np.zeros(0, dtype=np.float32)
        rms = _rms_envelope(audio, sr)
        return cls(
            sample_rate=sr,
            audio=audio,
            spans=spans,
            segments=segments,
            rms=rms,
            rms_hop=max(1, int(sr * 0.01)),
            language=timelines[0].language,
        )

    @classmethod
    def from_alignments(
        cls,
        audio: np.ndarray,
        sample_rate: int,
        alignments: Iterable[object],
        *,
        language: str = "zh",
    ) -> "Timeline":
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        spans: list[PhonemeSpan] = []
        segments: list[VisemeSegment] = []
        cursor = 0
        pending_marks = 0
        last_index: int | None = None

        def add_span(phone: str, count: int) -> None:
            nonlocal cursor, last_index
            if count <= 0:
                return
            span = PhonemeSpan(phone, cursor, cursor + count)
            spans.append(span)
            spec: VisemeSpec = map_phoneme(span.phoneme, language)
            segments.append(
                VisemeSegment(
                    span.start,
                    span.end,
                    span.phoneme,
                    spec.viseme,
                    spec.kind,
                    spec.open,
                    spec.form,
                    spec.vowel_gain,
                )
            )
            cursor = span.end
            last_index = len(segments) - 1

        for item in alignments:
            phoneme = getattr(item, "phoneme", None)
            count = getattr(item, "num_samples", None)
            if phoneme is None or count is None:
                continue
            phone = str(phoneme)
            count = int(count)
            if count <= 0:
                continue
            if phone in _TRANSPARENT_PHONES:
                # Stress/length/tone markers are timing metadata, not mouth
                # shapes. Carry their duration into the next real phone so the
                # mouth does not snap shut for 50 ms at every stressed syllable.
                pending_marks += count
                continue
            if phone in _SILENCE_PHONES:
                add_span("_", count + pending_marks)
                pending_marks = 0
                continue
            add_span(phone, count + pending_marks)
            pending_marks = 0

        if pending_marks:
            if last_index is not None:
                last = segments[last_index]
                segments[last_index] = VisemeSegment(
                    last.start,
                    last.end + pending_marks,
                    last.phoneme,
                    last.viseme,
                    last.kind,
                    last.open,
                    last.form,
                    last.vowel_gain,
                )
                cursor += pending_marks
            else:
                add_span("_", pending_marks)

        # eSpeak sometimes emits an epenthetic ɜ after o in a Chinese syllable
        # (e.g. 婆 -> o + ɜ). It is not a second visible vowel; merge it into
        # the rounded vowel so the mouth does not add an E shape mid-syllable.
        merged: list[VisemeSegment] = []
        for segment in segments:
            if (
                merged
                and segment.phoneme == "ɜ"
                and merged[-1].phoneme in {"o", "ɔ"}
                and segment.start - merged[-1].end <= int(sample_rate * 0.035)
            ):
                previous = merged[-1]
                merged[-1] = VisemeSegment(
                    previous.start,
                    segment.end,
                    previous.phoneme,
                    previous.viseme,
                    previous.kind,
                    max(previous.open, segment.open),
                    previous.form,
                    previous.vowel_gain,
                )
                continue
            merged.append(segment)
        segments = merged
        if cursor != audio.size:
            raise ValueError(f"alignment samples {cursor} != audio samples {audio.size}")
        rms = _rms_envelope(audio, sample_rate)
        return cls(
            sample_rate=sample_rate,
            audio=audio,
            spans=spans,
            segments=segments,
            rms=rms,
            rms_hop=max(1, int(sample_rate * 0.01)),
            language=language,
        )

    @classmethod
    def synthetic(
        cls,
        text: str,
        *,
        sample_rate: int = 48000,
        language: str = "zh",
        phone_ms: int = 95,
    ) -> "Timeline":
        """A silent but realistic-enough timeline for VTS-only testing.

        It is intentionally not a speech recognizer; it just gives the mixer a
        sequence of plausible mouth shapes so the VTS plumbing can be tested
        before a TTS voice is installed.
        """
        phones: list[str] = []
        for char in text:
            if char in "，。！？；：,.!?;:":
                phones.append("_")
            elif char.isspace():
                phones.append("_")
            elif language == "zh":
                phones.extend(_pinyin_like_phones(char))
            else:
                phones.append(char.lower())
        if not phones:
            phones = ["_"]
        count = max(1, int(sample_rate * phone_ms / 1000))
        audio = np.zeros(count * len(phones), dtype=np.float32)
        spans = [PhonemeSpan(phone, i * count, (i + 1) * count) for i, phone in enumerate(phones)]
        segments = []
        for span in spans:
            spec = map_phoneme(span.phoneme, language)
            segments.append(
                VisemeSegment(
                    span.start,
                    span.end,
                    span.phoneme,
                    spec.viseme,
                    spec.kind,
                    spec.open,
                    spec.form,
                    spec.vowel_gain,
                )
            )
        return cls(
            sample_rate=sample_rate,
            audio=audio,
            spans=spans,
            segments=segments,
            rms=np.zeros(max(1, audio.size // max(1, int(sample_rate * 0.01))), dtype=np.float32),
            rms_hop=max(1, int(sample_rate * 0.01)),
            language=language,
        )


def _silence_segment(start: int, end: int) -> VisemeSegment:
    return VisemeSegment(start, end, "_", "", "silence", 0.0, 0.0, 0.0)


def _rms_envelope(audio: np.ndarray, sample_rate: int, window_ms: float = 25.0) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return np.zeros(0, dtype=np.float32)
    hop = max(1, int(sample_rate * 0.01))
    window = max(1, int(sample_rate * window_ms / 1000.0))
    power = np.square(audio, dtype=np.float64)
    kernel = np.ones(window, dtype=np.float64) / window
    smoothed = np.convolve(power, kernel, mode="same")
    # Sample the smoothed power every 10 ms.
    indexes = np.arange(0, audio.size, hop, dtype=np.int64)
    values = np.sqrt(np.maximum(smoothed[indexes], 0.0)).astype(np.float32)
    return values


def _pinyin_like_phones(char: str) -> list[str]:
    """Very small Hanzi -> plausible phone helper for silent demos.

    This is not a pronunciation dictionary. For real speech Piper supplies
    the phones; this path exists only so a no-voice install can still verify
    the VTS parameter stream.
    """
    code = ord(char)
    if 0x4E00 <= code <= 0x9FFF:
        table = "a o e i u v ai ei ao ou an en ang eng ong".split()
        return [table[code % len(table)]]
    if char.isalpha():
        return [char.lower()]
    return ["_"]
