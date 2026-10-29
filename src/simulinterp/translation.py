"""Offline German/English to simplified Chinese translation.

The int8 M2M100 model runs on the CPU, leaving the laptop's VRAM for
speech recognition. No dependency import or model download occurs at import time.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import re
import threading
from typing import Any


_ABBREVIATIONS = re.compile(
    r"\b(?:mr|mrs|ms|dr|prof|sr|jr|st|mt|rev|gen|rep|sen|gov|sgt|capt|lt|col|"
    r"no|vs|etc|fig|approx|dept|inc|ltd|co|corp|hr|hrn|fr|dipl|ing|med|jur|"
    r"bzw|ca|usw|ggf|inkl|zzgl|gem|str|nr|abs|art|kap|bd|aufl|sog|evtl|vgl)\.",
    re.IGNORECASE,
)
_INITIALISMS = re.compile(r"(?<!\w)(?:[A-Za-zÄÖÜäöü]\.\s*){2,}")
_BOUNDARIES = re.compile(r"[.!?。！？]+[\"'”’»›)\]]*(?=\s|$)")


def split_sentences(text: str) -> list[str]:
    """Split punctuated EN/DE speech without breaking common abbreviations.

    This conservative heuristic protects titles, spaced abbreviations such as
    ``z. B.``, initialisms such as ``U.S.``, personal initials, and decimals.
    It does not attempt to infer sentence boundaries in unpunctuated speech.
    """
    protected = set()
    for pattern in (_ABBREVIATIONS, _INITIALISMS):
        for match in pattern.finditer(text):
            protected.update(i for i in range(match.start(), match.end()) if text[i] == ".")
    for match in re.finditer(r"(?<=\d)\.(?=\d)|\b[A-ZÄÖÜ]\.(?=\s+[A-ZÄÖÜ])", text):
        protected.add(match.end() - 1)
    sentences = []
    start = 0
    for match in _BOUNDARIES.finditer(text):
        if text[match.start()] == "." and match.start() in protected:
            continue
        # An ellipsis within a clause is not an extra sentence.
        following = text[match.end():].lstrip()
        if text[match.start():match.end()].startswith("..") and following[:1].islower():
            continue
        sentence = text[start:match.end()].strip()
        if sentence:
            sentences.append(sentence)
        start = match.end()
    remainder = text[start:].strip()
    if remainder:
        sentences.append(remainder)
    return sentences


class TranslationError(RuntimeError):
    """An actionable local translation failure."""


class LocalTranslator:
    """Translate with a locally converted CTranslate2 M2M100 model.

    ``model_dir`` contains ``model.bin`` and the tokenizer files copied by
    ``ct2-transformers-converter``. Tokenizer assets can alternatively be stored
    in ``tokenizer_dir``. A single instance is safe to call from multiple threads;
    requests are serialized because the tokenizer's source language is mutable.
    """

    def __init__(
        self,
        model_dir: str | Path,
        tokenizer_dir: str | Path | None = None,
        cpu_threads: int | None = None,
        beam_size: int = 1,
    ) -> None:
        if cpu_threads is None:
            from .config import DEFAULT_TRANSLATION_THREADS
            cpu_threads = DEFAULT_TRANSLATION_THREADS
        if cpu_threads < 1 or beam_size < 1:
            raise ValueError("cpu_threads 和 beam_size 必须大于 0")
        self.model_dir = Path(model_dir)
        self.tokenizer_dir = Path(tokenizer_dir) if tokenizer_dir else self.model_dir
        self.cpu_threads = cpu_threads
        self.beam_size = beam_size
        self._translator: Any = None
        self._tokenizer: Any = None
        self._simplifier: Any = None
        self._lock = threading.RLock()
        self._cache: OrderedDict[tuple[str, str], str] = OrderedDict()

    @property
    def loaded(self) -> bool:
        return self._translator is not None

    def close(self) -> None:
        """Release inference and tokenizer references after any active request."""
        with self._lock:
            self._translator = None
            self._tokenizer = None
            self._simplifier = None
            self._cache.clear()

    def load(self) -> None:
        """Load only local assets. Missing files never trigger a network request."""
        with self._lock:
            if self.loaded:
                return
            if not (self.model_dir / "model.bin").is_file():
                raise TranslationError(
                    f"未找到本地翻译模型：{self.model_dir}。请先运行模型下载/准备脚本。"
                )
            try:
                import ctranslate2
                from opencc import OpenCC
                from transformers import M2M100Tokenizer
            except ImportError as exc:
                raise TranslationError("翻译依赖未安装完整，请重新运行安装脚本。") from exc
            try:
                tokenizer = M2M100Tokenizer.from_pretrained(
                    str(self.tokenizer_dir), local_files_only=True
                )
                translator = ctranslate2.Translator(
                    str(self.model_dir),
                    device="cpu",
                    compute_type="int8",
                    inter_threads=1,
                    intra_threads=self.cpu_threads,
                )
                simplifier = OpenCC("t2s")
            except Exception as exc:
                raise TranslationError(
                    f"无法加载本地翻译模型，请检查模型和分词器是否完整：{exc}"
                ) from exc
            self._tokenizer = tokenizer
            self._simplifier = simplifier
            # Set last so a failed load cannot appear successful.
            self._translator = translator

    def translate(self, text: str, source_language: str) -> str:
        """Translate a finalized ASR segment, returning simplified Chinese.

        Sentences are translated independently to reduce dropped source clauses.
        Long sentences are split in token space instead of silently truncated.
        Only ``en`` and ``de`` are accepted, matching the application's audio
        language controls. Empty input does not load either model.
        """
        if source_language not in {"en", "de"}:
            raise ValueError("只支持英文 en 或德文 de")
        text = text.strip()
        if not text:
            return ""
        key = (source_language, text)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            self.load()
            tokenizer = self._tokenizer
            tokenizer.src_lang = source_language
            try:
                # Special tokens must be included explicitly for CT2 models
                # converted from Transformers (see the official CT2 guide).
                target_token = tokenizer.lang_code_to_token["zh"]
                translated_sentences: list[str] = []
                for sentence in split_sentences(text):
                    token_ids = tokenizer.encode(sentence, add_special_tokens=False)
                    translated_parts: list[str] = []
                    for offset in range(0, len(token_ids), 384):
                        chunk = tokenizer.build_inputs_with_special_tokens(
                            token_ids[offset : offset + 384]
                        )
                        source_tokens = tokenizer.convert_ids_to_tokens(chunk)
                        results = self._translator.translate_batch(
                            [source_tokens],
                            target_prefix=[[target_token]],
                            beam_size=self.beam_size,
                            return_scores=False,
                            max_input_length=0,
                            max_decoding_length=768,
                        )
                        if not results or not results[0].hypotheses:
                            raise TranslationError("翻译模型未返回结果")
                        output_tokens = results[0].hypotheses[0]
                        if len(output_tokens) >= 768:
                            raise TranslationError("翻译结果超过长度限制，请缩短语音分段。")
                        if output_tokens and output_tokens[0] == target_token:
                            output_tokens = output_tokens[1:]
                        decoded = tokenizer.decode(
                            tokenizer.convert_tokens_to_ids(output_tokens),
                            skip_special_tokens=True,
                        ).strip()
                        if not decoded:
                            raise TranslationError("翻译模型返回空文本")
                        translated_parts.append(decoded)
                    translated_sentence = "".join(translated_parts)
                    if not translated_sentence:
                        raise TranslationError("翻译模型返回空文本")
                    translated_sentences.append(translated_sentence)
                # Chinese needs no inter-sentence spaces. Supply a full stop
                # only when the model omitted terminal punctuation entirely.
                for index in range(len(translated_sentences) - 1):
                    if not translated_sentences[index].rstrip('\"\'”’»›)]').endswith(tuple("。！？.!?…")):
                        translated_sentences[index] = translated_sentences[index].rstrip("，,；;：:") + "。"
                translation = self._simplifier.convert("".join(translated_sentences)).strip()
                if not translation:
                    raise TranslationError("翻译模型返回空文本")
            except TranslationError:
                raise
            except Exception as exc:
                raise TranslationError(f"本地翻译失败：{exc}") from exc
            self._cache[key] = translation
            if len(self._cache) > 128:
                self._cache.popitem(last=False)
            return translation
