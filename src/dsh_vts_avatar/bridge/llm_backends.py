"""Pluggable local/remote LLM backends.

Two implementations cover the useful cases:

- ``ollama``: POST /api/chat on a local Ollama server (default);
- ``openai``: any OpenAI-compatible /chat/completions endpoint (LM Studio,
  llama.cpp server, vLLM, a hosted API...);
- ``none``: disabled, every call returns an empty string.

The bridge only talks to ``generate``; prompt building and validation stay in
``local_chat.py`` so switching backends cannot change her behaviour.
"""

from __future__ import annotations

import asyncio
import json
import urllib.request
from typing import Any

from ..config import LLMConfig


class LLMBackend:
    name = "base"

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float = 0.8,
        num_predict: int = 140,
        timeout: float = 60.0,
    ) -> str:
        raise NotImplementedError

    def _post(self, url: str, payload: dict[str, Any], timeout: float, headers: dict[str, str] | None = None) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json", **(headers or {})},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))


class OllamaBackend(LLMBackend):
    name = "ollama"

    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint.rstrip("/")

    async def generate(self, messages, *, model, temperature=0.8, num_predict=140, timeout=60.0) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {"temperature": temperature, "num_predict": num_predict},
        }
        data = await asyncio.to_thread(
            self._post, self.endpoint + "/api/chat", payload, timeout
        )
        return str((data.get("message") or {}).get("content") or "").strip()


class OpenAICompatBackend(LLMBackend):
    name = "openai"

    def __init__(self, endpoint: str, api_key: str = "") -> None:
        base = endpoint.rstrip("/")
        if not base.endswith("/v1"):
            base = base + "/v1"
        self.endpoint = base
        self.api_key = api_key

    async def generate(self, messages, *, model, temperature=0.8, num_predict=140, timeout=60.0) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": num_predict,
            "stream": False,
        }
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = await asyncio.to_thread(
            self._post,
            self.endpoint + "/chat/completions",
            payload,
            timeout,
            headers,
        )
        choices = data.get("choices") or []
        if not choices:
            return ""
        return str((choices[0].get("message") or {}).get("content") or "").strip()


class NullBackend(LLMBackend):
    name = "none"

    async def generate(self, messages, *, model, temperature=0.8, num_predict=140, timeout=60.0) -> str:
        return ""


def make_backend(config: LLMConfig) -> LLMBackend:
    kind = str(config.backend or "ollama").strip().lower()
    if kind in ("none", "off", "disabled"):
        return NullBackend()
    if kind in ("openai", "openai-compatible", "compatible"):
        return OpenAICompatBackend(config.endpoint, config.api_key)
    return OllamaBackend(config.endpoint)
