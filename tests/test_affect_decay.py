from dsh_vts_avatar.config import AvatarConfig
from dsh_vts_avatar.performance.affect import AffectState, compose_face
from dsh_vts_avatar.performance.mixer import PerformanceMixer


def _mixer(state: AffectState) -> PerformanceMixer:
    mixer = PerformanceMixer(AvatarConfig(mouth_backend="continuous"), 48000, use_affect=True)
    mixer.set_affect(state)
    return mixer


def test_affect_holds_then_fades():
    state = AffectState(labels={"happy": 1.0}, intensity=1.0, confidence=0.9)
    mixer = _mixer(state)
    smile_before = compose_face(state, 1.0)["smile"]

    # Inside the hold window nothing moves.
    for _ in range(10):
        mixer._update_affect(0.033)
    assert state.intensity == 1.0

    # After the hold, the described state decays instead of freezing the face.
    mixer._affect_stamp -= mixer.AFFECT_HOLD_S + 1.5
    for _ in range(60):
        mixer._update_affect(0.033)
    assert state.intensity < 0.7
    assert compose_face(state, 1.0)["smile"] < smile_before


def test_new_evidence_restarts_the_hold():
    state = AffectState(labels={"sad": 1.0}, intensity=0.8, confidence=0.9)
    mixer = _mixer(state)
    mixer._affect_stamp -= mixer.AFFECT_HOLD_S + 3.0
    for _ in range(30):
        mixer._update_affect(0.033)
    faded = state.intensity
    assert faded < 0.8

    mixer.set_affect(state)  # a fresh feel must not keep decaying immediately
    mixer._update_affect(0.033)
    assert state.intensity == faded
