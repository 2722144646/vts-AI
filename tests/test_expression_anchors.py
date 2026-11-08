from dsh_vts_avatar.performance.affect import anchor, is_known
from dsh_vts_avatar.performance.affect.composer import compose_face
from dsh_vts_avatar.performance.affect.schema import AffectState


def test_nuanced_labels_are_known():
    for label in ("suppressed", "doubtful", "guilty", "relieved"):
        assert is_known(label)
    assert anchor("suppressed")[0] < 0


def test_suppressed_is_not_neutral():
    face = compose_face(AffectState(labels={"suppressed": 1.0}, intensity=1.0))
    assert face["smile"] < 0.4
    assert face["inner_brow"] > 0.3


def test_chinese_aliases_map_to_nuanced_labels():
    assert anchor("压抑") == anchor("suppressed")
    assert anchor("半信半疑") == anchor("doubtful")
