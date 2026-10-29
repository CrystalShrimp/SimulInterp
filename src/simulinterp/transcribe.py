"""Turn a local German/English recording into a local SRT, fully offline.

Example: python -m simulinterp.transcribe --audio lecture.mp4 --language de
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator
from pathlib import Path
import tempfile
import time

import numpy as np

from .config import (DATA_ROOT, DEFAULT_ASR_PROFILE, DEFAULT_TRANSLATION_THREADS,
                     TRANSLATION_THREAD_OPTIONS, ASR_PROFILES, asr_path, model_status,
                     segmentation_options, translation_path)
from .subtitles import Subtitle, render_srt


def audio_blocks(path: Path) -> Iterator[np.ndarray]:
    """Decode a local recording a frame at a time into mono 16 kHz float audio.

    PyAV ships with faster-whisper. Streaming avoids holding an entire lecture
    in RAM and works for both audio files and the audio track of local videos.
    """
    import av

    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise ValueError("文件中没有可用的音轨")
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="flt", layout="mono", rate=16000)
        for frame in container.decode(stream):
            for output in resampler.resample(frame):
                yield output.to_ndarray().reshape(-1).astype(np.float32, copy=False)
        for output in resampler.resample(None):
            yield output.to_ndarray().reshape(-1).astype(np.float32, copy=False)


def transcribe_file(
    audio: Path,
    *,
    language: str,
    device: str = "cuda",
    model_size: str = "large-v3-turbo",
    translation_threads: int = DEFAULT_TRANSLATION_THREADS,
    translate_enabled: bool = True,
    asr_profile: str = DEFAULT_ASR_PROFILE,
    progress: Callable[[str], None] = print,
) -> list[Subtitle]:
    if language not in ("de", "en"):
        raise ValueError("必须指定课程语言：de 或 en")
    if type(translate_enabled) is not bool:
        raise ValueError("是否翻译必须为布尔值")
    if translate_enabled and translation_threads not in TRANSLATION_THREAD_OPTIONS:
        raise ValueError("翻译 CPU 线程数必须为 4、6 或 8")
    phrase_options = segmentation_options(asr_profile)
    from .runtime import bootstrap

    bootstrap(offline=True)
    from .recognition import LocalRecognizer
    from .segmenter import SpeechChunk, SpeechSegmenter

    recognizer = LocalRecognizer(asr_path(model_size), device)
    translator = None
    if translate_enabled:
        from .translation import LocalTranslator

        translator = LocalTranslator(translation_path(), cpu_threads=translation_threads)
    segmenter = SpeechSegmenter(**phrase_options)
    entries: list[Subtitle] = []

    def process(chunk: SpeechChunk) -> None:
        transcript = recognizer.transcribe(chunk.samples, language)
        if transcript is None:
            return
        display_text = translator.translate(transcript.text, transcript.language) if translator is not None else transcript.text
        if not display_text.strip():
            return
        start = min(chunk.end, chunk.start + max(0.0, transcript.start))
        end = min(chunk.end, chunk.start + max(0.0, transcript.end))
        entries.append(Subtitle(start, max(start, end), transcript.text, display_text, transcript.language))
        progress(f"[{start:.1f}s] {display_text}")

    try:
        progress("正在加载本地语音识别与翻译模型 …" if translate_enabled else "正在加载本地语音识别模型（仅原文字幕） …")
        recognizer.load()
        if translator is not None:
            translator.load()
        for samples in audio_blocks(audio):
            for chunk in segmenter.feed(samples):
                process(chunk)
        for chunk in segmenter.flush():
            process(chunk)
        return entries
    finally:
        recognizer.close()
        if translator is not None:
            translator.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True, help="本地音频/视频路径")
    parser.add_argument("--language", choices=("de", "en"), required=True, help="课程源语言；不执行自动检测")
    parser.add_argument("--translation-threads", type=int, choices=TRANSLATION_THREAD_OPTIONS,
                        default=DEFAULT_TRANSLATION_THREADS, help="中文翻译 CPU 线程数")
    parser.add_argument("--no-translation", action="store_true", help="仅输出德语 / 英语原文字幕，不加载翻译模型")
    parser.add_argument("--asr-profile", choices=tuple(ASR_PROFILES), default=DEFAULT_ASR_PROFILE,
                        help="断句节奏：adaptive 清晰收音（默认），coherent 较差收音长上下文，fast 低延迟")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--model", choices=("large-v3-turbo", "small"), default="large-v3-turbo")
    parser.add_argument("--output", type=Path, default=DATA_ROOT / "exports" / "subtitles.srt")
    args = parser.parse_args(argv)
    if not args.audio.is_file():
        parser.error(f"音频文件不存在：{args.audio}")
    if args.audio.resolve() == args.output.resolve():
        parser.error("字幕输出路径必须与音频输入路径不同")
    ready, message = model_status(args.model, translate_enabled=False) if args.no_translation else model_status(args.model)
    if not ready:
        print(message + "；请先运行 python -m simulinterp.prepare_models", flush=True)
        return 1
    temporary_path: Path | None = None
    started = time.monotonic()
    try:
        entries = transcribe_file(
            args.audio,
            language=args.language,
            device=args.device,
            model_size=args.model,
            translation_threads=args.translation_threads,
            translate_enabled=not args.no_translation,
            asr_profile=args.asr_profile,
            progress=lambda message: print(message, flush=True),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Publish only a complete SRT; failed conversion preserves any existing
        # subtitles at the requested destination.
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", delete=False,
            prefix=".subtitles-", suffix=".tmp", dir=args.output.parent,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(render_srt(entries))
        temporary_path.replace(args.output)
        temporary_path = None
        subtitle_kind = "原文字幕" if args.no_translation else "中文字幕"
        print(f"已导出 {len(entries)} 条{subtitle_kind}：{args.output.resolve()}（处理耗时 {time.monotonic() - started:.1f}s）", flush=True)
        return 0
    except KeyboardInterrupt:
        print("已中断，未保存未完成的字幕。", flush=True)
        return 130
    except Exception as exc:
        print(f"文件转译失败：{exc}", flush=True)
        return 1
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
