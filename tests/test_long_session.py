"""Accelerated 90-minute stream checks, without devices or neural inference.

The real segmenter sees all 86,400,000 samples in bounded input blocks. This
regression checks duration/state/export behavior, not a 90-minute hardware soak.
"""

import numpy as np
import pytest

from simulinterp.config import segmentation_options
from simulinterp.segmenter import SpeechSegmenter
from simulinterp.subtitles import Subtitle, render_srt, render_text


class _SpeechVad:
    def is_speech(self, pcm: bytes, sample_rate: int) -> bool:
        return True


class _MeasuredSegmenter(SpeechSegmenter):
    def __init__(self, **options):
        super().__init__(vad=_SpeechVad(), **options)
        self.peak_active_frames = 0
        self.peak_pre_roll_frames = 0
        self.peak_pending_samples = 0

    def _process_frame(self, frame):
        result = super()._process_frame(frame)
        self.peak_active_frames = max(self.peak_active_frames, len(self._active))
        self.peak_pre_roll_frames = max(self.peak_pre_roll_frames, len(self._pre_roll))
        self.peak_pending_samples = max(self.peak_pending_samples, self._pending.size)
        return result


@pytest.mark.parametrize("profile,max_seconds,expected_count,last_start", [
    ("adaptive", 8, 675, "01:29:52,000"),
    ("coherent", 8, 675, "01:29:52,000"),
    ("fast", 5, 1080, "01:29:55,000"),
])
def test_ninety_minutes_streams_in_bounded_state_and_exports_all_captions(profile, max_seconds, expected_count, last_start):
    sample_rate = 16000
    seconds = 90 * 60
    block_seconds = 5
    # Reuse a single block; the test never constructs a recording-sized array.
    block = np.full(sample_rate * block_seconds, 0.2, dtype=np.float32)
    segmenter = _MeasuredSegmenter(**segmentation_options(profile))
    entries = []
    total_output_samples = 0
    previous_end = 0.0

    for _ in range(seconds // block_seconds):
        chunks = segmenter.feed(block)
        for chunk in chunks:
            assert chunk.start == previous_end
            assert 0 < chunk.duration <= max_seconds
            assert chunk.samples.size == round(chunk.duration * sample_rate)
            total_output_samples += chunk.samples.size
            previous_end = chunk.end
            # The app retains text metadata for the full class, not PCM arrays.
            index = len(entries) + 1
            entries.append(Subtitle(chunk.start, chunk.end, f"phrase {index}",
                                    f"第 {index} 段字幕", "en"))

    assert segmenter.flush() == []
    assert len(entries) == expected_count
    assert total_output_samples == sample_rate * seconds
    assert previous_end == seconds
    assert segmenter._cursor == sample_rate * seconds
    assert segmenter._pending.size == 0
    assert len(segmenter._active) == 0
    assert len(segmenter._pre_roll) == 0
    assert segmenter.peak_active_frames <= max_seconds * 50  # 20 ms frames
    assert segmenter.peak_pre_roll_frames <= 11  # configured pre-roll
    assert segmenter.peak_pending_samples <= sample_rate * block_seconds

    srt = render_srt(entries)
    text = render_text(entries)
    assert f"01:00:00,000 --> 01:00:{max_seconds:02d},000" in srt
    assert f"{expected_count}\n{last_start} --> 01:30:00,000\n第 {expected_count} 段字幕" in srt
    assert srt.count(" --> ") == expected_count
    assert f"[{last_start}] 第 {expected_count} 段字幕\nphrase {expected_count}" in text
    # A complete class is small text; no need to drop old exportable captions.
    assert len(srt.encode("utf-8")) < 256 * 1024
    assert len(text.encode("utf-8")) < 256 * 1024
