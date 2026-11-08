from dsh_vts_avatar.performance.relationship import RelationshipStore


def test_events_move_the_numbers_asymmetrically(tmp_path):
    store = RelationshipStore(tmp_path, "u1")
    trust0 = store.state.trust
    store.apply("promise_broken", now=1000.0)
    assert store.state.trust < trust0
    assert store.state.friction > 0
    store.apply("apology_accepted", now=1000.0)
    assert store.state.friction < 0.1


def test_friction_decays_but_trust_stays_longer(tmp_path):
    store = RelationshipStore(tmp_path, "u1")
    store.apply("conflict", now=1000.0)
    friction = store.state.friction
    trust = store.state.trust
    store.decay(now=1000.0 + 48 * 3600.0)
    assert store.state.friction < friction / 2
    assert store.state.trust > trust - 0.02


def test_impression_only_when_it_matters(tmp_path):
    store = RelationshipStore(tmp_path, "u1")
    assert store.impression() == ""
    store.apply("conflict", now=1000.0)
    store.apply("conflict", now=1000.0)
    store.apply("conflict", now=1000.0)
    assert "摩擦" in store.impression() or "余波" in store.impression()


def test_state_survives_reload(tmp_path):
    store = RelationshipStore(tmp_path, "u1")
    store.apply("vulnerability_shared", now=1000.0)
    reloaded = RelationshipStore(tmp_path, "u1")
    assert reloaded.state.closeness > 0.2
