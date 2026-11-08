import asyncio

from dsh_vts_avatar.bridge.engine import AvatarEngine
from dsh_vts_avatar.config import Config, EmotionConfig, LLMConfig


class StubLocal:
    async def reply(self, text, history, extra_system=""):
        return "嗯，我在。"


def make_engine(tmp_path):
    config = Config(root=tmp_path)
    config.emotion = EmotionConfig(mode="affect")
    config.llm = LLMConfig(enabled=True, history_turns=2)
    engine = AvatarEngine(config)
    engine.local_chat = StubLocal()

    async def noop(*args, **kwargs):
        return None

    engine.begin_utterance = noop
    engine.feed_text = noop
    engine.end_utterance = noop
    return engine


def test_local_reply_hides_text_by_default(tmp_path):
    engine = make_engine(tmp_path)
    result = asyncio.run(engine.local_reply("hi"))
    assert result["ok"] and "reply" not in result


def test_local_reply_can_show_text(tmp_path):
    engine = make_engine(tmp_path)
    result = asyncio.run(engine.local_reply("hi", show_text=True))
    assert result["reply"] == "嗯，我在。"


def test_save_llm_settings_patches_config(tmp_path):
    (tmp_path / "config.toml").write_text('[llm]\nbackend = "ollama"\nmodel = "a"\n', encoding="utf-8")
    engine = make_engine(tmp_path)
    result = engine.save_llm_settings({"backend": "openai", "model": "b", "evil": "x"})
    assert result["ok"] and result["restart_required"]
    body = (tmp_path / "config.toml").read_text(encoding="utf-8")
    assert 'backend = "openai"' in body and 'model = "b"' in body
    assert "evil" not in body
