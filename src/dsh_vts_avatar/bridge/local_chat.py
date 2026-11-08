"""Prompting for the private speech path and the idle director.

The actual transport lives in ``llm_backends.py``; this module only builds
messages, validates output, and keeps the character/user rules. That way
switching from Ollama to an OpenAI-compatible endpoint does not change how
she behaves.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..config import LLMConfig
from .llm_backends import LLMBackend, make_backend


def _parse_json_object(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


class LocalChat:
    """The avatar's own voice, generated on this machine or a configured API."""

    def __init__(self, config: LLMConfig, backend: LLMBackend | None = None) -> None:
        self.config = config
        self.backend = backend or make_backend(config)

    async def reply(
        self, user_text: str, history: list[dict[str, str]], extra_system: str = ""
    ) -> str:
        system = (self.config.persona or "").strip()
        if extra_system:
            system = (system + "\n\n" + extra_system).strip()
        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        limit = max(2, int(self.config.history_turns) * 2)
        messages.extend(
            {"role": m["role"], "content": m["content"]}
            for m in history[-limit:]
            if m.get("role") in ("user", "assistant") and m.get("content")
        )
        messages.append({"role": "user", "content": user_text})
        return await self.backend.generate(
            messages,
            model=self.config.model,
            temperature=0.8,
            num_predict=140,
            timeout=self.config.timeout_s,
        )


class LocalDirector:
    """Asks the model to pick the next idle motion; None means fall back."""

    def __init__(self, config: LLMConfig, backend: LLMBackend | None = None) -> None:
        self.config = config
        self.backend = backend or make_backend(config)

    async def suggest(
        self, kinds: list[str], context: dict[str, Any]
    ) -> tuple[str, float] | None:
        prompt = (
            "你是 Live2D 角色的待机导演。根据下面的状态，从候选动作里挑一个最自然的，"
            "只输出 JSON，不要解释：{\"kind\":\"候选之一\",\"intensity\":0.1-1.2}\n"
            "尽量避开 recent 里刚做过的动作，除非完全没有别的选择。\n"
            f"状态：{json.dumps(context, ensure_ascii=False)}\n"
            f"候选：{', '.join(kinds)}"
        )
        messages = [
            {"role": "system", "content": "你只输出一行 JSON。"},
            {"role": "user", "content": prompt},
        ]
        try:
            text = await self.backend.generate(
                messages,
                model=self.config.director_model,
                temperature=0.7,
                num_predict=100,
                timeout=self.config.director_timeout_s,
            )
        except Exception:
            return None
        obj = _parse_json_object(text)
        if obj is None:
            return None
        kind = str(obj.get("kind") or obj.get("action") or obj.get("name") or "")
        if kind not in kinds:
            return None
        try:
            intensity = float(obj.get("intensity", 0.5))
        except (TypeError, ValueError):
            intensity = 0.5
        return kind, round(min(1.2, max(0.1, intensity)), 3)
