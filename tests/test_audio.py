from library_escape.audio import (
    available_audio_files,
    grade_for_score,
    missing_legacy_audio_files,
    score_sound_key,
)


def test_repo_audio_files_match_legacy_cpp_assets_that_still_exist():
    files = available_audio_files()
    assert "bgm" in files
    assert "detected" in files
    assert "score_a" in files
    assert "score_f" in files


def test_missing_collect_and_powerup_assets_are_explicitly_reported():
    missing = missing_legacy_audio_files()
    assert "collect" in missing
    assert "powerup" in missing


def test_grade_mapping_and_score_sound_selection_are_stable():
    assert grade_for_score(72, 72) == "A+"
    assert grade_for_score(65, 72) == "A"
    assert grade_for_score(44, 72) == "C"
    assert grade_for_score(10, 72) == "F"
    assert score_sound_key("A+") == "score_aplus"
    assert score_sound_key("B-") == "score_bminus"
    assert score_sound_key("F") == "score_f"
