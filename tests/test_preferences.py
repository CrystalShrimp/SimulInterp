import json
from unittest.mock import patch

import pytest

from simulinterp.preferences import default_preferences, load_preferences, save_preferences


def test_first_use_and_old_auto_mode_require_an_explicit_language(tmp_path):
    path = tmp_path / "settings.json"
    assert load_preferences(path)["language"] is None
    path.write_text(json.dumps({"language": "auto", "translation_threads": 8}), encoding="utf-8")
    restored = load_preferences(path)
    assert restored["language"] is None
    assert restored["translation_threads"] == 8


def test_local_settings_round_trip_without_device_identifiers(tmp_path):
    path = tmp_path / "custom" / "settings.json"
    save_preferences({"language": "de", "model_size": "large-v3-turbo", "device": "cuda",
                      "translation_threads": 4, "device_id": 1234}, path)
    restored = load_preferences(path)
    assert restored == dict(language="de", model_size="large-v3-turbo", device="cuda",
                            asr_profile="adaptive", translate_enabled=True, translation_threads=4)
    assert "device_id" not in path.read_text(encoding="utf-8")
    assert not list(path.parent.glob(".settings-*.tmp"))


@pytest.mark.parametrize("contents", ["not json", "[]", "null", '{"language": [], "model_size": [], "device": {}, "translation_threads": true}'])
def test_invalid_or_corrupt_settings_fall_back_safely(tmp_path, contents):
    path = tmp_path / "settings.json"
    path.write_text(contents, encoding="utf-8")
    assert load_preferences(path) == default_preferences()


def test_atomic_publication_failure_keeps_previous_settings(tmp_path):
    path = tmp_path / "settings.json"
    save_preferences({"language": "en", "translation_threads": 8}, path)
    original = path.read_bytes()
    with patch("simulinterp.preferences.os.replace", side_effect=OSError("disk full")):
        with pytest.raises(OSError, match="disk full"):
            save_preferences({"language": "de"}, path)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob(".settings-*.tmp"))


@pytest.mark.parametrize("values", [{}, {"asr_profile": None}, {"asr_profile": "unknown"},
                                   {"asr_profile": []}, {"asr_profile": {}}, {"asr_profile": True}])
def test_missing_or_invalid_caption_timing_defaults_to_adaptive(tmp_path, values):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"language": "en", **values}), encoding="utf-8")
    restored = load_preferences(path)
    assert restored["asr_profile"] == "adaptive"
    assert restored["language"] == "en"


@pytest.mark.parametrize("profile", ["adaptive", "coherent", "fast"])
def test_explicit_caption_timing_persists_with_other_settings(tmp_path, profile):
    path = tmp_path / "settings.json"
    save_preferences({"language": "de", "asr_profile": profile, "translation_threads": 4}, path)
    restored = load_preferences(path)
    assert restored["asr_profile"] == profile
    assert restored["language"] == "de"
    assert restored["translation_threads"] == 4


@pytest.mark.parametrize("enabled", [True, False])
def test_translation_choice_persists_as_a_boolean(tmp_path, enabled):
    path = tmp_path / "settings.json"
    save_preferences({"language": "en", "translate_enabled": enabled}, path)
    assert load_preferences(path)["translate_enabled"] is enabled


@pytest.mark.parametrize("values", [{}, {"translate_enabled": None}, {"translate_enabled": 0},
                                   {"translate_enabled": "false"}, {"translate_enabled": []}])
def test_old_or_malformed_translation_choice_keeps_translation_enabled(tmp_path, values):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(values), encoding="utf-8")
    assert load_preferences(path)["translate_enabled"] is True


