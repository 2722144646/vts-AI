import asyncio

from dsh_vts_avatar.bridge.llm_backends import (
    NullBackend,
    OllamaBackend,
    OpenAICompatBackend,
    make_backend,
)
from dsh_vts_avatar.config import LLMConfig


class RecordingOllama(OllamaBackend):
    def __init__(self, endpoint="http://127.0.0.1:11434"):
        super().__init__(endpoint)
        self.calls = []

    def _post(self, url, payload, timeout, headers=None):
        self.calls.append({"url": url, "payload": payload, "timeout": timeout, "headers": headers})
        return {"message": {"content": "hello"}}


class RecordingOpenAI(OpenAICompatBackend):
    def __init__(self, endpoint="https://api.example.com", key="k"):
        super().__init__(endpoint, key)
        self.calls = []

    def _post(self, url, payload, timeout, headers=None):
        self.calls.append({"url": url, "payload": payload, "headers": headers})
        return {"choices": [{"message": {"content": "hi there"}}]}


def test_ollama_backend_payload_and_parse():
    backend = RecordingOllama()
    text = asyncio.run(
        backend.generate([{"role": "user", "content": "hi"}], model="m", timeout=5)
    )
    assert text == "hello"
    call = backend.calls[-1]
    assert call["url"].endswith("/api/chat")
    assert call["payload"]["model"] == "m"
    assert call["payload"]["stream"] is False


def test_openai_backend_payload_and_parse():
    backend = RecordingOpenAI()
    text = asyncio.run(
        backend.generate([{"role": "user", "content": "hi"}], model="m", timeout=5)
    )
    assert text == "hi there"
    call = backend.calls[-1]
    assert call["url"] == "https://api.example.com/v1/chat/completions"
    assert call["headers"]["Authorization"] == "Bearer k"
    assert call["payload"]["max_tokens"] == 140


def test_make_backend_dispatch():
    assert isinstance(make_backend(LLMConfig(backend="none")), NullBackend)
    assert isinstance(make_backend(LLMConfig(backend="ollama")), OllamaBackend)
    assert isinstance(make_backend(LLMConfig(backend="openai")), OpenAICompatBackend)
    # unknown names fall back to ollama instead of crashing
    assert isinstance(make_backend(LLMConfig(backend="whatever")), OllamaBackend)


def test_null_backend_returns_empty():
    assert asyncio.run(NullBackend().generate([], model="m")) == ""
