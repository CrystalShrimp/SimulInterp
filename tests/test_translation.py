"""Translation protocol tests; no network, weights, or ML packages needed."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from simulinterp.translation import LocalTranslator, TranslationError, split_sentences


class FakeTokenizer:
    lang_code_to_token = {"zh": "__zh__"}
    src_lang = None

    def encode(self, text, add_special_tokens=True):
        assert add_special_tokens is False
        return text.split()

    def build_inputs_with_special_tokens(self, ids):
        return [f"__{self.src_lang}__", *ids, "</s>"]

    def convert_ids_to_tokens(self, ids):
        return ids

    def convert_tokens_to_ids(self, tokens):
        return tokens

    def decode(self, ids, skip_special_tokens=False):
        assert skip_special_tokens is True
        return "".join(token for token in ids if token != "</s>")


class FakeEngine:
    def __init__(self):
        self.calls = []

    def translate_batch(self, sources, **kwargs):
        self.calls.append((sources, kwargs))
        return [SimpleNamespace(hypotheses=[["__zh__", "這是字幕。", "</s>"]])]


def prepared_translator():
    translator = LocalTranslator(Path("unused"))
    translator._tokenizer = FakeTokenizer()
    translator._translator = FakeEngine()
    translator._simplifier = SimpleNamespace(convert=lambda text: text.replace("這", "这").replace("</s>", ""))
    return translator


def test_empty_input_does_not_load_model(tmp_path):
    translator = LocalTranslator(tmp_path)
    assert translator.translate("  ", "en") == ""
    assert not translator.loaded


def test_missing_weights_have_actionable_error(tmp_path):
    with pytest.raises(TranslationError, match="模型"):
        LocalTranslator(tmp_path).translate("Hello", "en")


def test_source_language_and_target_prefix_and_simplification():
    translator = prepared_translator()
    assert translator.translate("Guten Tag", "de") == "这是字幕。"
    sources, options = translator._translator.calls[0]
    assert sources == [["__de__", "Guten", "Tag", "</s>"]]
    assert options["target_prefix"] == [["__zh__"]]
    assert options["max_input_length"] == 0
    assert options["beam_size"] == 1
    assert options["return_scores"] is False


def test_explicit_beam_override_and_threads_remain_available():
    translator = prepared_translator()
    translator.beam_size = 2
    translator.translate("Guten Tag", "de")
    assert translator._translator.calls[0][1]["beam_size"] == 2
    assert translator._translator.calls[0][1]["return_scores"] is False


def test_translation_threads_use_tuned_config_default(monkeypatch):
    import simulinterp.config as config
    monkeypatch.setattr(config, "DEFAULT_TRANSLATION_THREADS", 4)
    assert LocalTranslator("unused").cpu_threads == 4
    assert LocalTranslator("unused", cpu_threads=8).cpu_threads == 8


def test_language_is_part_of_cache_key():
    translator = prepared_translator()
    translator.translate("test", "en")
    translator.translate("test", "en")
    translator.translate("test", "de")
    assert len(translator._translator.calls) == 2
    assert translator._translator.calls[1][0][0][0] == "__de__"


def test_long_input_is_not_truncated():
    translator = prepared_translator()
    text = " ".join(f"word{i}" for i in range(800))
    translator.translate(text, "en")
    calls = translator._translator.calls
    assert len(calls) == 3
    reconstructed = [word for sources, _ in calls for word in sources[0][1:-1]]
    assert reconstructed == text.split()


def test_unsupported_language_is_rejected():
    with pytest.raises(ValueError, match="支持"):
        LocalTranslator("unused").translate("bonjour", "fr")


def test_empty_hypothesis_is_not_cached():
    translator = prepared_translator()
    translator._translator.translate_batch = lambda *args, **kwargs: [SimpleNamespace(hypotheses=[])]
    with pytest.raises(TranslationError, match="未返回"):
        translator.translate("Hello", "en")
    assert len(translator._cache) == 0


def test_close_releases_models_and_cache():
    translator = prepared_translator()
    translator.translate("Hello", "en")
    translator.close()
    assert not translator.loaded
    assert translator._tokenizer is None
    assert translator._simplifier is None
    assert not translator._cache


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Good morning. Today we discuss technology.", ["Good morning.", "Today we discuss technology."]),
        ("Guten Morgen! Heute sprechen wir über Sprachen.", ["Guten Morgen!", "Heute sprechen wir über Sprachen."]),
        ("Dr. Müller spricht. Wir hören zu.", ["Dr. Müller spricht.", "Wir hören zu."]),
        ("Das ist z. B. beim Lernen wichtig. Es hilft.", ["Das ist z. B. beim Lernen wichtig.", "Es hilft."]),
        ("He works in the U.S. economy. That matters.", ["He works in the U.S. economy.", "That matters."]),
        ("The value is 3.14. Is that correct? Yes!", ["The value is 3.14.", "Is that correct?", "Yes!"]),
        ("J. Smith is speaking. Please listen.", ["J. Smith is speaking.", "Please listen."]),
        ('He said "Welcome!" Then he left.', ['He said "Welcome!"', "Then he left."]),
        ("Wait... what happened? Explain.", ["Wait... what happened?", "Explain."]),
        ("  No punctuation here  ", ["No punctuation here"]),
        ("\n \t", []),
    ],
)
def test_sentence_boundaries_preserve_abbreviations_and_decimals(text, expected):
    assert split_sentences(text) == expected


def test_every_source_sentence_is_translated_separately_in_order():
    translator = prepared_translator()
    text = "Good morning. Today we are talking about language learning and technology."
    assert translator.translate(text, "en") == "这是字幕。这是字幕。"
    sources = [source[0][1:-1] for source, _ in translator._translator.calls]
    assert sources == [
        ["Good", "morning."],
        ["Today", "we", "are", "talking", "about", "language", "learning", "and", "technology."],
    ]


def test_sentence_join_supplies_missing_terminal_punctuation():
    translator = prepared_translator()
    outputs = iter(["第一句", "第二句。"])
    translator._translator.translate_batch = lambda *args, **kwargs: [
        SimpleNamespace(hypotheses=[["__zh__", next(outputs)]])
    ]
    assert translator.translate("First sentence. Second sentence.", "en") == "第一句。第二句。"


def test_long_sentence_chunking_still_preserves_all_following_sentences():
    translator = prepared_translator()
    long_sentence = " ".join(f"word{i}" for i in range(800)) + "."
    translator.translate(long_sentence + " Last sentence.", "en")
    calls = translator._translator.calls
    assert len(calls) == 4
    recovered = [word for sources, _ in calls for word in sources[0][1:-1]]
    assert recovered == (long_sentence + " Last sentence.").split()
