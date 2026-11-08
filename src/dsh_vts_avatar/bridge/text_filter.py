"""Turn streamed Markdown into text that is safe to send to TTS.

The avatar should speak the answer, not read code fences, JSON, tables or
inline code.  This filter is stateful because a code fence can be split across
arbitrary stream deltas.
"""

from __future__ import annotations

import re


class SpeechTextFilter:
    def __init__(self) -> None:
        self._in_fence = False
        self._pending = ""

    def reset(self) -> None:
        self._in_fence = False
        self._pending = ""

    def feed(self, text: str) -> str:
        if not text:
            return ""
        data = self._pending + text
        self._pending = ""
        out: list[str] = []
        i = 0
        while i < len(data):
            if self._in_fence:
                end = data.find("```", i)
                if end < 0:
                    return self._clean("".join(out))
                i = end + 3
                self._in_fence = False
                continue
            start = data.find("```", i)
            if start < 0:
                tail = data[i:]
                # Hold one/two trailing backticks: they may be the beginning
                # of a fence split across two deltas.
                match = re.search(r"`{1,2}$", tail)
                if match:
                    out.append(tail[: match.start()])
                    self._pending = match.group(0)
                else:
                    out.append(tail)
                break
            out.append(data[i:start])
            i = start + 3
            self._in_fence = True
        return self._clean("".join(out))

    def flush(self) -> str:
        pending = self._pending
        self._pending = ""
        self._in_fence = False
        return self._clean(pending.replace("`", ""))

    @staticmethod
    def _clean(text: str) -> str:
        # Inline code often contains identifiers that should not be spoken.
        text = re.sub(r"`[^`\n]*`", "", text)
        text = text.replace("`", "")
        lines: list[str] = []
        for line in text.splitlines():
            if line.strip().startswith("|"):
                continue
            line = re.sub(r"^\s{0,3}#{1,6}\s*", "", line)
            line = re.sub(r"[*_]{1,3}", "", line)
            lines.append(line)
        return "\n".join(lines)
