import asyncio

from dsh_vts_avatar.bridge.local_chat import LocalDirector
from dsh_vts_avatar.config import LLMConfig


class FakeBackend:
    def __init__(self, content):
        self.content = content
        self.calls = []

    async def generate(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return self.content


def director_with(content):
    backend = FakeBackend(content)
    director = LocalDirector(
        LLMConfig(director_enabled=True, director_model="m"), backend=backend
    )
    return director, backend


def test_director_parses_json():
    director, _ = director_with('{"kind": "idle_tilt", "intensity": 0.8}')
    assert asyncio.run(director.suggest(["idle_tilt", "idle_nod"], {})) == ("idle_tilt", 0.8)


def test_director_accepts_action_alias():
    director, _ = director_with('{"action": "idle_glance"}')
    assert asyncio.run(director.suggest(["idle_glance"], {})) == ("idle_glance", 0.5)


def test_director_strips_code_fence():
    director, _ = director_with('```json\n{"kind": "idle_nod", "intensity": 0.4}\n```')
    assert asyncio.run(director.suggest(["idle_nod"], {})) == ("idle_nod", 0.4)


def test_director_rejects_unknown_kind():
    director, _ = director_with('{"kind": "idle_dance", "intensity": 0.5}')
    assert asyncio.run(director.suggest(["idle_tilt"], {})) is None


def test_director_rejects_bad_json():
    director, _ = director_with("sorry, I cannot")
    assert asyncio.run(director.suggest(["idle_tilt"], {})) is None


def test_director_clamps_intensity():
    director, _ = director_with('{"kind": "idle_tilt", "intensity": 9}')
    assert asyncio.run(director.suggest(["idle_tilt"], {})) == ("idle_tilt", 1.2)


def test_director_uses_director_model_and_timeout():
    director, backend = director_with('{"kind": "idle_tilt"}')
    asyncio.run(director.suggest(["idle_tilt"], {}))
    _, kwargs = backend.calls[-1]
    assert kwargs["model"] == "m"
    assert kwargs["timeout"] == director.config.director_timeout_s
