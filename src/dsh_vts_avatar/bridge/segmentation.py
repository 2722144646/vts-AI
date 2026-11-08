"""Stream-friendly sentence segmentation.

The first segment is deliberately short (latency), later segments are longer
(prosody). Tags are stripped before TTS but parsed for emotion by the caller.
"""

from __future__ import annotations

import re

STRONG_END = set("。！？!?；;\n")
WEAK_END = set("，,、：:")
SPACE = set(" \t\r\n")

_TAG_RE = re.compile(r"\[[^\]]{0,64}\]")


def strip_tags(text: str) -> str:
    return _TAG_RE.sub("", text)


def _last_break(text: str, max_chars: int) -> int:
    window = text[:max_chars]
    for i in range(len(window) - 1, -1, -1):
        if window[i] in WEAK_END:
            return i + 1
    for i in range(len(window) - 1, -1, -1):
        if window[i] in SPACE:
            return i + 1
    return max_chars


class Segmenter:
    def __init__(self, *, first_max: int = 30, max_chars: int = 72):
        self.first_max = first_max
        self.max_chars = max_chars
        self.buffer = ""
        self.emitted = 0

    def feed(self, text: str) -> list[str]:
        self.buffer += text
        out: list[str] = []
        while True:
            segment = self._take()
            if segment is None:
                break
            if segment.strip():
                out.append(segment.strip())
                self.emitted += 1
        return out

    def flush(self) -> list[str]:
        out: list[str] = []
        if self.buffer.strip():
            out.append(self.buffer.strip())
        self.buffer = ""
        return out

    def _take(self) -> str | None:
        limit = self.first_max if self.emitted == 0 else self.max_chars
        # For the first phrase, a clause boundary is more useful than waiting
        # for a full sentence: it cuts the first audible response time.
        if self.emitted == 0:
            for index, char in enumerate(self.buffer):
                if char in WEAK_END and index + 1 >= 3:
                    return self._cut(index + 1)
        # Prefer a strong end once we have enough material for natural prosody.
        for index, char in enumerate(self.buffer):
            if char in STRONG_END and index + 1 >= 6:
                return self._cut(index + 1)
        if len(self.buffer) >= limit:
            cut = _last_break(self.buffer, limit)
            return self._cut(cut)
        return None

    def _cut(self, index: int) -> str:
        segment = self.buffer[:index]
        self.buffer = self.buffer[index:]
        return segment.strip()
