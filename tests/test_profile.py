from pathlib import Path

from dsh_vts_avatar.vts.profile import ModelProfile, load_for_model, profile_path


def test_profiles_are_per_model_and_keep_tuning(tmp_path: Path):
    a = ModelProfile(model_name="A", model_id="model-a", tuning={"head_gain": 1.6})
    b = ModelProfile(model_name="B", model_id="model-b", tuning={"head_gain": 2.8})
    a.save(profile_path(tmp_path, a.model_id))
    b.save(profile_path(tmp_path, b.model_id))

    assert profile_path(tmp_path, "model-a") != profile_path(tmp_path, "model-b")
    loaded_a = load_for_model(tmp_path, "model-a")
    loaded_b = load_for_model(tmp_path, "model-b")
    assert loaded_a.tuning["head_gain"] == 1.6
    assert loaded_b.tuning["head_gain"] == 2.8
    assert loaded_a.model_name == "A"
    assert loaded_b.model_name == "B"
