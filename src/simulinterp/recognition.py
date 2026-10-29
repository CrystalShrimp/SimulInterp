from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

@dataclass(frozen=True)
class Transcript:
    text: str
    language: str
    start: float
    end: float


class LocalRecognizer:
    def __init__(self, model_dir: Path, device: str = "cuda"):
        if device not in ("cuda", "cpu"):
            raise ValueError("device must be cuda or cpu")
        self.model_dir = model_dir
        self.device = device
        self._model = None

    def load(self) -> None:
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            str(self.model_dir), device=self.device,
            compute_type="int8_float16" if self.device == "cuda" else "int8",
            cpu_threads=6, num_workers=1, local_files_only=True,
        )

    def warmup(self) -> None:
        """Initialize GPU kernels before capture starts, without producing text.

        faster-whisper executes encoder/decoder work when its lazy segment
        generator is consumed. Disable VAD and silence filtering so silent
        warmup audio still exercises that path. CPU sessions need no GPU warmup.
        """
        if self.device != "cuda":
            return
        if self._model is None:
            raise RuntimeError("识别模型尚未加载")
        segments, _ = self._model.transcribe(
            np.zeros(16000, dtype=np.float32),
            language="en", task="transcribe", beam_size=1, temperature=0.0,
            condition_on_previous_text=False, vad_filter=False,
            without_timestamps=True, max_new_tokens=1,
            no_speech_threshold=None, log_prob_threshold=None,
            compression_ratio_threshold=None,
        )
        for _ in segments:
            pass

    def transcribe(self, samples: np.ndarray, language: str) -> Transcript | None:
        if language not in ("de", "en"):
            raise ValueError("请先指定课程语言：德语 de 或英语 en；不使用自动语言检测")
        if self._model is None:
            raise RuntimeError("识别模型尚未加载")
        segments, _ = self._model.transcribe(
            samples, language=language,
            task="transcribe", beam_size=3, temperature=0.0,
            condition_on_previous_text=False, vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
            no_speech_threshold=0.6, log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4, word_timestamps=False,
        )
        kept = [s for s in segments if s.text.strip() and not (s.no_speech_prob > .6 and s.avg_logprob < -1)]
        if not kept:
            return None
        text = " ".join(s.text.strip() for s in kept).strip()
        return Transcript(text, language, kept[0].start, kept[-1].end)

    def close(self) -> None:
        self._model = None
