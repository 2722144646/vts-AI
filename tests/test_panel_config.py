from dsh_vts_avatar.tools.panel import read_section, set_section_value


def test_set_updates_key_and_keeps_comments(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[llm]\n# keep me\nbackend = "ollama"\nmodel = "a"\n\n[other]\nbackend = "x"\n', encoding="utf-8")
    text = set_section_value(path.read_text(encoding="utf-8"), "llm", "backend", "openai")
    path.write_text(text, encoding="utf-8")
    assert read_section(path, "llm")["backend"] == "openai"
    body = path.read_text(encoding="utf-8")
    assert "# keep me" in body
    assert '[other]\nbackend = "x"' in body


def test_set_adds_missing_key(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[llm]\nmodel = "a"\n', encoding="utf-8")
    text = set_section_value(path.read_text(encoding="utf-8"), "llm", "api_key", "sk-1")
    path.write_text(text, encoding="utf-8")
    assert read_section(path, "llm")["api_key"] == "sk-1"


def test_set_creates_missing_section(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[audio]\nx = 1\n', encoding="utf-8")
    text = set_section_value(path.read_text(encoding="utf-8"), "llm", "backend", "none")
    path.write_text(text, encoding="utf-8")
    assert read_section(path, "llm")["backend"] == "none"


def test_boolean_roundtrip(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[llm]\ndirector_enabled = false\n', encoding="utf-8")
    text = set_section_value(path.read_text(encoding="utf-8"), "llm", "director_enabled", True)
    path.write_text(text, encoding="utf-8")
    assert read_section(path, "llm")["director_enabled"] == "true"
