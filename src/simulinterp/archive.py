"""Append completed Chinese captions to a recoverable local SRT per session."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4

from .subtitles import Subtitle, timestamp


class SessionArchive:
    """Keep no in-memory history and flush every accepted subtitle to the OS.

    This survives a process crash; it is not a guarantee against power failure
    or a failing disk. The controller retains text separately for manual export.
    """

    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        name = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
        self.path = directory / f"{name}.srt"
        self._stream = self.path.open("x", encoding="utf-8-sig", newline="\n")
        self.count = 0

    def append(self, item: Subtitle) -> None:
        index = self.count + 1
        self._stream.write(
            f"{index}\n{timestamp(item.start)} --> "
            f"{timestamp(max(item.start + .001, item.end))}\n{item.text.strip()}\n\n"
        )
        self._stream.flush()
        self.count = index

    def close(self) -> None:
        self._stream.close()


def list_session_archives(directory: Path) -> list[dict]:
    """Summaries of past class SRT files, newest first, for the cleanup UI."""
    if not directory.is_dir():
        return []
    sessions = []
    for path in directory.glob("*.srt"):
        try:
            stat = path.stat()
        except OSError:
            continue
        sessions.append({"path": str(path), "name": path.stem,
                         "size_bytes": stat.st_size,
                         "modified": datetime.fromtimestamp(stat.st_mtime)})
    # The archive name starts with %Y%m%d-%H%M%S, so name order is time order.
    sessions.sort(key=lambda item: item["name"], reverse=True)
    return sessions
