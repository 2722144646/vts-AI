from dsh_vts_avatar.bridge.segmentation import Segmenter


def test_first_segment_is_short_and_punctuation_kept():
    seg = Segmenter(first_max=12, max_chars=30)
    out = seg.feed("你好，我是 dsh。今天想和你聊聊。")
    assert out
    assert out[0].endswith("，") or out[0].endswith("。")
    assert len(out[0]) <= 12


def test_flush_returns_remainder():
    seg = Segmenter()
    assert seg.feed("没有标点的一句话") == []
    assert seg.flush() == ["没有标点的一句话"]
