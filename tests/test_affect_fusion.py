from dsh_vts_avatar.performance.affect import (
    SOURCE_WEIGHTS,
    AffectEvidence,
    AffectFusion,
    AffectState,
)


def _clock():
    now = [100.0]
    return now, (lambda: now[0])


def test_sources_blend_instead_of_being_ranked():
    now, clock = _clock()
    fusion = AffectFusion(clock=clock)
    fusion.add(
        AffectEvidence(source="tool", labels={"happy": 1.0}, intensity=0.8, confidence=0.9, timestamp=now[0])
    )
    state = fusion.add(
        AffectEvidence(source="keyword", labels={"sad": 1.0}, intensity=0.6, confidence=0.5, timestamp=now[0])
    )

    # Conflict stays visible as a mixed state; nothing is thrown away.
    assert set(state.labels) == {"happy", "sad"}
    assert state.labels["happy"] > state.labels["sad"]
    assert state.source == "fusion"
    assert -0.2 < state.valence < 0.8


def test_manual_is_a_vote_and_tool_still_leads():
    now, clock = _clock()
    fusion = AffectFusion(clock=clock)
    fusion.add(
        AffectEvidence(source="tool", labels={"happy": 1.0}, confidence=0.9, timestamp=now[0])
    )
    state = fusion.add(
        AffectEvidence(source="manual", labels={"sad": 1.0}, confidence=1.0, timestamp=now[0])
    )

    # No source hard-overrides: the command is a 0.5-weight vote, and the AI's
    # own reading (weight 1.0) still leads.
    assert set(state.labels) == {"happy", "sad"}
    assert state.labels["happy"] > state.labels["sad"]
    assert state.source == "fusion"


def test_manual_alone_still_moves_the_face():
    now, clock = _clock()
    fusion = AffectFusion(clock=clock)
    state = fusion.add(
        AffectEvidence(source="manual", labels={"sad": 1.0}, intensity=0.8, confidence=1.0, timestamp=now[0])
    )
    assert state.labels == {"sad": 1.0}
    assert state.intensity > 0.5


def test_manual_expires_like_any_other_evidence():
    now, clock = _clock()
    fusion = AffectFusion(clock=clock)
    fusion.add(
        AffectEvidence(source="manual", labels={"sad": 1.0}, confidence=1.0, timestamp=now[0])
    )
    now[0] += 11.0  # past evidence_ttl_s
    state = fusion.state()
    assert state.labels == {"neutral": 1.0}


def test_evidence_expires_to_neutral():
    now, clock = _clock()
    fusion = AffectFusion(clock=clock)
    fusion.add(
        AffectEvidence(source="tool", labels={"happy": 1.0}, intensity=0.9, timestamp=now[0])
    )
    now[0] += 11.0  # past evidence_ttl_s
    state = fusion.state()
    assert state.intensity == 0.0
    assert state.labels == {"neutral": 1.0}


def test_keyword_vote_weighs_less_than_tool_vote():
    now, clock = _clock()
    fusion = AffectFusion(clock=clock)
    fusion.add(AffectEvidence(source="keyword", labels={"sad": 1.0}, confidence=0.9, timestamp=now[0]))
    tool_state = fusion.add(
        AffectEvidence(source="tool", labels={"happy": 1.0}, confidence=0.9, timestamp=now[0])
    )
    assert tool_state.labels["happy"] > tool_state.labels["sad"]


def test_ndi_never_votes():
    now, clock = _clock()
    fusion = AffectFusion(clock=clock)
    fusion.add(AffectEvidence(source="ndi", labels={"angry": 1.0}, confidence=1.0, timestamp=now[0]))
    state = fusion.state()
    assert state.labels == {"neutral": 1.0}
    assert state.intensity == 0.0


def test_empty_fusion_is_neutral():
    state = AffectFusion().state()
    assert state.labels == {"neutral": 1.0}
    assert state.intensity == 0.0


def test_tool_self_report_has_the_highest_weight():
    # The AI's own reading is the only source that saw the conversation.
    assert max(SOURCE_WEIGHTS.values()) == SOURCE_WEIGHTS["tool"]
    assert SOURCE_WEIGHTS["tool"] > SOURCE_WEIGHTS["tag"]
    assert SOURCE_WEIGHTS["tool"] > SOURCE_WEIGHTS["keyword"]
    # A user command is just a half-weight vote; nothing hard-overrides.
    assert SOURCE_WEIGHTS["manual"] == 0.5
    assert SOURCE_WEIGHTS["manual"] < SOURCE_WEIGHTS["tool"]
