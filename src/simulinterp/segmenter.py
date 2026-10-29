"""Bounded speech segmentation with silence endpointing and absolute timestamps."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Protocol

import numpy as np


class VoiceDetector(Protocol):
    def is_speech(self, pcm: bytes, sample_rate: int) -> bool: ...


@dataclass(frozen=True)
class SpeechChunk:
    samples: np.ndarray
    start: float
    end: float
    # Capture-stream position when the endpoint was detected, not wall time.
    # Kept separate from end, which trims most of the endpointing silence.
    available_at: float | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start


class SpeechSegmenter:
    """Split mono float audio into phrases or bounded continuous-speech chunks.

    VAD runs on 20 ms frames. The 200 ms pre-roll preserves quiet consonants;
    a 120 ms trailing margin avoids chopping word endings. All timestamps are
    measured in the original capture stream, including discarded silence.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        max_seconds: float = 5.0,
        silence_seconds: float = 0.5,
        min_speech_seconds: float = 0.18,
        pre_roll_seconds: float = 0.2,
        vad: VoiceDetector | None = None,
        soft_min_seconds: float | None = None,
        soft_silence_seconds: float | None = None,
    ):
        if sample_rate not in (8000, 16000, 32000, 48000):
            raise ValueError("WebRTC VAD requires 8/16/32/48 kHz audio")
        if max_seconds < 0.5 or silence_seconds <= 0 or min_speech_seconds < 0:
            raise ValueError("Invalid segmentation durations")
        if min_speech_seconds > max_seconds:
            raise ValueError("Minimum speech cannot exceed maximum chunk duration")
        if not 0 <= pre_roll_seconds < max_seconds:
            raise ValueError("Pre-roll must fit inside a chunk")
        if (soft_min_seconds is None) != (soft_silence_seconds is None):
            raise ValueError("Adaptive endpoint durations must be provided together")
        if soft_min_seconds is not None:
            try:
                valid_soft = (math.isfinite(soft_min_seconds) and math.isfinite(soft_silence_seconds)
                              and 0 < soft_min_seconds < max_seconds
                              and 0 < soft_silence_seconds < silence_seconds)
            except TypeError:
                valid_soft = False
            if not valid_soft:
                raise ValueError("Adaptive endpoint durations must be positive and fit inside the normal limits")
        if vad is None:
            import webrtcvad

            vad = webrtcvad.Vad(2)
        self.sample_rate = sample_rate
        self._vad = vad
        self._frame_size = sample_rate // 50
        self._max_samples = math.floor(max_seconds * sample_rate / self._frame_size) * self._frame_size
        self._silence_limit = round(silence_seconds * sample_rate)
        self._soft_min = max(1, math.ceil(soft_min_seconds * sample_rate)) if soft_min_seconds is not None else None
        self._soft_silence_limit = (max(1, math.ceil(soft_silence_seconds * sample_rate))
                                    if soft_silence_seconds is not None else None)
        self._min_speech = round(min_speech_seconds * sample_rate)
        self._trail = round(0.12 * sample_rate)
        self._pre_roll = deque(maxlen=min(
            self._max_samples // self._frame_size,
            max(1, math.ceil(pre_roll_seconds * 50) + 1),
        ))
        self._pending = np.empty(0, dtype=np.float32)
        self._cursor = 0
        self._active: list[np.ndarray] = []
        self._active_start = 0
        self._voiced_samples = 0
        self._last_voice_end = 0
        self._silence_samples = 0

    def feed(self, samples: np.ndarray) -> list[SpeechChunk]:
        audio = np.asarray(samples, dtype=np.float32)
        if audio.ndim != 1:
            raise ValueError("Segmenter requires mono audio")
        if not np.all(np.isfinite(audio)):
            raise ValueError("Audio contains non-finite samples")
        self._pending = np.concatenate((self._pending, np.clip(audio, -1.0, 1.0)))
        output = []
        while self._pending.size >= self._frame_size:
            frame = self._pending[:self._frame_size].copy()
            self._pending = self._pending[self._frame_size:]
            chunk = self._process_frame(frame)
            if chunk is not None:
                output.append(chunk)
        return output

    def _process_frame(self, frame: np.ndarray) -> SpeechChunk | None:
        start = self._cursor
        self._cursor += frame.size
        padded = np.pad(frame, (0, self._frame_size - frame.size))
        pcm = (np.clip(padded, -1.0, 1.0) * 32767).astype("<i2").tobytes()
        voiced = self._vad.is_speech(pcm, self.sample_rate)
        self._pre_roll.append((start, frame))
        if not self._active:
            if not voiced:
                return None
            self._active_start = self._pre_roll[0][0]
            self._active = [item[1] for item in self._pre_roll]
        else:
            self._active.append(frame)
        if voiced:
            self._voiced_samples += frame.size
            self._last_voice_end = self._cursor
            self._silence_samples = 0
        else:
            self._silence_samples += frame.size
        forced = self._cursor - self._active_start >= self._max_samples
        silence_limit = self._silence_limit
        # Count retained context through the last voiced frame, not the quiet
        # wait itself. A short utterance must not earn early endpointing merely
        # by sitting in silence. Continuous speech still uses the hard cap.
        if self._soft_min is not None and self._last_voice_end - self._active_start >= self._soft_min:
            silence_limit = self._soft_silence_limit
        if forced or self._silence_samples >= silence_limit:
            result = self._emit()
            if forced:
                self._pre_roll.clear()
            return result
        return None

    def _emit(self) -> SpeechChunk | None:
        result = None
        if self._active and self._voiced_samples >= self._min_speech:
            end = min(self._cursor, self._last_voice_end + self._trail)
            samples = np.concatenate(self._active)[:end - self._active_start].copy()
            result = SpeechChunk(samples=samples, start=self._active_start / self.sample_rate,
                                 end=end / self.sample_rate, available_at=self._cursor / self.sample_rate)
        self._active = []
        self._voiced_samples = 0
        self._silence_samples = 0
        return result

    def flush(self) -> list[SpeechChunk]:
        """Emit pending speech at stop; calling flush twice is harmless."""
        output = []
        if self._pending.size:
            remainder = self._pending
            self._pending = np.empty(0, dtype=np.float32)
            chunk = self._process_frame(remainder)
            if chunk is not None:
                output.append(chunk)
        chunk = self._emit()
        if chunk is not None:
            output.append(chunk)
        self._pre_roll.clear()
        return output
