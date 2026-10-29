import pytest

from simulinterp.subtitles import Subtitle, render_srt, render_text, timestamp


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0, "00:00:00,000"),
        (-5.3, "00:00:00,000"),
        (1.234, "00:00:01,234"),
        (59.9996, "00:01:00,000"),
        (3661.234, "01:01:01,234"),
        (360000, "100:00:00,000"),
    ],
)
def test_timestamp(seconds, expected):
    assert timestamp(seconds) == expected


def test_srt_keeps_chinese_and_uses_absolute_times():
    entries = [
        Subtitle(61.25, 63.75, "Guten Morgen.", "  早上好。  ", "de"),
        Subtitle(65, 66, "Welcome.", "欢迎。", "en"),
    ]
    assert render_srt(entries) == (
        "1\n00:01:01,250 --> 00:01:03,750\n早上好。\n\n"
        "2\n00:01:05,000 --> 00:01:06,000\n欢迎。\n"
    )


@pytest.mark.parametrize("end", [2, 1])
def test_srt_ensures_positive_duration(end):
    entry = Subtitle(2, end, "Hello", "你好", "en")
    assert "00:00:02,000 --> 00:00:02,001" in render_srt([entry])


def test_empty_exports_are_empty_strings():
    assert render_srt([]) == ""
    assert render_text([]) == ""


def test_text_export_preserves_source_for_review():
    entry = Subtitle(5.25, 7, "Guten Tag.", "你好。", "de")
    assert render_text([entry]) == "[00:00:05,250] 你好。\nGuten Tag.\n"


@pytest.mark.parametrize("language,text", [("en", "The lecture is starting."), ("de", "Die Vorlesung beginnt.")])
def test_source_only_exports_show_one_copy_of_original(language, text):
    entry = Subtitle(0, 2, text, text, language)
    assert render_srt([entry]) == f"1\n00:00:00,000 --> 00:00:02,000\n{text}\n"
    assert render_text([entry]) == f"[00:00:00,000] {text}\n"
