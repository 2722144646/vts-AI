from dsh_vts_avatar.bridge.text_filter import SpeechTextFilter


def test_code_fence_is_not_spoken():
    f = SpeechTextFilter()
    text = f.feed("你好。\n```json\n{\"a\": 1}\n```\n再见。")
    assert "你好" in text
    assert "再见" in text
    assert "json" not in text
    assert "{" not in text
    assert "```" not in text


def test_fence_split_across_deltas():
    f = SpeechTextFilter()
    out = f.feed("前面。``")
    assert out == "前面。"
    out += f.feed("`json\n{}\n`")
    assert "json" not in out
    out += f.feed("```后面。")
    assert "后面" in out


def test_inline_code_is_removed():
    f = SpeechTextFilter()
    out = f.feed("参数 `MouthOpen` 现在可用。")
    assert "MouthOpen" not in out
    assert "参数" in out and "现在可用" in out
