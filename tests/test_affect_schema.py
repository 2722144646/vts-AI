from dsh_vts_avatar.performance.affect import (
    AffectEvidence,
    AffectLog,
    AffectState,
    derive_valence_arousal,
    normalize_labels,
    read_log,
)


def test_labels_are_clamped_and_renormalized():
    labels = normalize_labels({"Happy": 2.0, "tired": 0.4, "sad": 0.0, "": 0.9})
    assert set(labels) == {"happy", "tired"}
    assert abs(sum(labels.values()) - 1.0) < 1e-9
    # 2.0 clamps to 1.0 before renormalizing, so happy is 1/1.4 not 2/2.4.
    assert abs(labels["happy"] - 1.0 / 1.4) < 1e-9
    assert labels["tired"] < labels["happy"]


def test_empty_labels_fall_back_to_neutral():
    assert normalize_labels({}) == {"neutral": 1.0}
    assert normalize_labels({"sad": 0.0}) == {"neutral": 1.0}
    assert normalize_labels(None) == {"neutral": 1.0}


def test_anchors_drive_valence_and_arousal():
    happy_v, happy_a = derive_valence_arousal({"happy": 1.0})
    sad_v, _ = derive_valence_arousal({"sad": 1.0})
    neutral_v, neutral_a = derive_valence_arousal({"neutral": 1.0})
    assert happy_v > 0.5 and sad_v < -0.5
    assert happy_a > neutral_a
    assert abs(neutral_v) < 1e-9
    assert abs(neutral_a - 0.15) < 1e-9


def test_mixed_state_stays_in_between():
    happy_v, happy_a = derive_valence_arousal({"happy": 1.0})
    mixed_v, mixed_a = derive_valence_arousal(normalize_labels({"happy": 0.6, "tired": 0.4}))
    assert 0.0 < mixed_v < happy_v
    assert mixed_a < happy_a


def test_evidence_validates_source_and_ranges():
    evidence = AffectEvidence(source="tool", labels={"happy": 1.0}, intensity=3.0, confidence=-1.0)
    assert evidence.intensity == 1.0
    assert evidence.confidence == 0.0
    assert evidence.is_weak
    try:
        AffectEvidence(source="vibes", labels={"happy": 1.0})
    except ValueError as exc:
        assert "vibes" in str(exc)
    else:  # pragma: no cover - the assertion below is the real check
        raise AssertionError("unknown source must be rejected")


def test_state_from_evidence_keeps_dominant_label():
    evidence = AffectEvidence(source="tag", labels={"happy": 0.6, "tired": 0.4}, intensity=0.7, confidence=0.9)
    state = AffectState.from_evidence(evidence)
    assert state.dominant()[0][0] == "happy"
    assert 0.0 < state.valence < 1.0
    assert state.confidence == 0.9
    assert state.labels == normalize_labels({"happy": 0.6, "tired": 0.4})


def test_neutral_state_is_not_a_strong_emotion():
    state = AffectState.neutral()
    assert state.is_neutral()
    assert state.intensity == 0.0
    assert abs(state.valence) < 1e-9


def test_log_round_trips(tmp_path):
    log = AffectLog(tmp_path / "affect")
    evidence = AffectEvidence(source="keyword", labels={"sad": 1.0}, note="test")
    log.evidence(evidence)
    log.state(AffectState.from_evidence(evidence))
    log.event("switch", {"from": "happy", "to": "sad"})

    files = list((tmp_path / "affect").glob("affect-*.jsonl"))
    assert len(files) == 1
    rows = read_log(files[0])
    assert [row["kind"] for row in rows] == ["evidence", "state", "switch"]
    assert rows[0]["labels"] == {"sad": 1.0}
    assert rows[2]["from"] == "happy"


def test_disabled_log_writes_nothing(tmp_path):
    log = AffectLog(tmp_path / "affect", enabled=False)
    log.evidence(AffectEvidence(source="keyword", labels={"happy": 1.0}))
    assert not (tmp_path / "affect").exists()
    assert log.failures == 0


def test_log_failure_never_raises(tmp_path):
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory", encoding="utf-8")
    log = AffectLog(blocker / "affect")
    log.state(AffectState.neutral())
    assert log.failures == 1
