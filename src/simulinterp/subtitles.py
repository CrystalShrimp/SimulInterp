from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Subtitle:
    start: float
    end: float
    source: str
    text: str
    language: str


def timestamp(seconds: float) -> str:
    total = max(0, round(seconds * 1000))
    hours, total = divmod(total, 3_600_000)
    minutes, total = divmod(total, 60_000)
    secs, millis = divmod(total, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def render_srt(entries: list[Subtitle]) -> str:
    return "\n\n".join(
        f"{index}\n{timestamp(item.start)} --> {timestamp(max(item.start + .001, item.end))}\n{item.text.strip()}"
        for index, item in enumerate(entries, 1)
    ) + ("\n" if entries else "")


def render_text(entries: list[Subtitle]) -> str:
    return "\n\n".join(
        f"[{timestamp(item.start)}] {item.text}" +
        (f"\n{item.source}" if item.source.strip() and item.source.strip() != item.text.strip() else "")
        for item in entries
    ) + ("\n" if entries else "")
