from dsh_vts_avatar.tools.chat_viewer import load_rows


def test_load_rows_reads_jsonl_and_skips_bad_lines(tmp_path):
    path = tmp_path / "chat.jsonl"
    path.write_text(
        '{"at": "2026-11-08T10:00:00", "user": "a", "assistant": "b"}\n'
        "not json\n"
        '{"at": "2026-11-08T10:01:00", "user": "c", "assistant": "d"}\n',
        encoding="utf-8",
    )
    rows = load_rows(path)
    assert len(rows) == 2
    assert rows[0]["assistant"] == "b"
    assert rows[1]["user"] == "c"


def test_load_rows_missing_file_is_empty(tmp_path):
    assert load_rows(tmp_path / "nope.jsonl") == []
