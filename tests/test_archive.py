from unittest.mock import Mock
import pytest

from simulinterp.archive import SessionArchive, list_session_archives
from simulinterp.subtitles import Subtitle


def test_subtitles_survive_without_close_and_after_an_hour(tmp_path):
    archive = SessionArchive(tmp_path / "sessions")
    archive.append(Subtitle(5398, 5400, "source", "最后一句", "de"))
    # A separate reader already sees flushed text while a class is running.
    assert archive.path.read_text(encoding="utf-8-sig") == (
        "1\n01:29:58,000 --> 01:30:00,000\n最后一句\n\n"
    )
    archive.append(Subtitle(5401, 5402, "next", "课程继续", "en"))
    archive.close()
    assert "2\n01:30:01,000" in archive.path.read_text(encoding="utf-8-sig")


def test_two_sessions_never_overwrite_and_keep_no_caption_history(tmp_path):
    a, b = SessionArchive(tmp_path), SessionArchive(tmp_path)
    try:
        assert a.path != b.path
        for n in range(1080):
            a.append(Subtitle(n * 5, n * 5 + 5, "English", "中文字幕", "en"))
        assert a.count == 1080
        assert "01:30:00,000" in a.path.read_text(encoding="utf-8-sig")
        assert not any(isinstance(value, (list, dict)) for value in vars(a).values())
        assert b.path.read_bytes() == b""
    finally:
        a.close()
        b.close()


def test_disk_write_failure_preserves_previous_complete_records(tmp_path):
    archive = SessionArchive(tmp_path)
    archive.append(Subtitle(0, 1, "source", "已经保存", "en"))
    original = archive.path.read_bytes()
    archive.close()
    archive._stream = Mock()
    archive._stream.write.side_effect = OSError("disk full")
    with pytest.raises(OSError, match="disk full"):
        archive.append(Subtitle(0, 1, "source", "字幕", "en"))
    assert archive.count == 1
    assert archive.path.read_bytes() == original


def test_list_session_archives_is_newest_first_and_reports_each_file(tmp_path):
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    (sessions / "20261027-080000-aaaa1111.srt").write_text("1\n00:00,000", encoding="utf-8",
                                                           newline="\n")
    (sessions / "20261028-093000-bbbb2222.srt").write_text("1\n00:00,000\n2\n00:05,000",
                                                           encoding="utf-8", newline="\n")
    (sessions / "notes.txt").write_text("not a session", encoding="utf-8")
    listed = list_session_archives(sessions)
    assert [item["name"] for item in listed] == ["20261028-093000-bbbb2222", "20261027-080000-aaaa1111"]
    assert listed[0]["path"].endswith("bbbb2222.srt")
    assert listed[0]["size_bytes"] == 23
    assert listed[0]["modified"] is not None
    assert list_session_archives(tmp_path / "missing") == []
