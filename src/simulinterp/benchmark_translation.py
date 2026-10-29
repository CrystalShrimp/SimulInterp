"""Sequential local CPU translation tuning; never loads speech/GPU models.

Run: python -m simulinterp.benchmark_translation --output cpu-tuning.json
The corpus is authored for this project, not a quality benchmark. Results describe
warm inference under the machine's current background load, not a long-class SLA.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import time

from .runtime import bootstrap


CORPUS = (
    ("en", "Today we discuss language learning and technology."),
    ("de", "Heute besprechen wir die Grundlagen der Elektrotechnik."),
    ("en", "Please compare these two methods and explain why the second experiment produced a different result."),
    ("de", "Wenn die Temperatur steigt, bewegen sich die Teilchen schneller und der Druck nimmt zu."),
    ("en", "The probability depends on the available information and the assumptions used in the model."),
    ("de", "Diese Gleichung beschreibt den Zusammenhang zwischen Energie, Masse und Geschwindigkeit."),
)


def _uncached_translate(translator, text, language):
    # Inference, rather than the application's 128-entry LRU, is under test.
    with translator._lock:
        translator._cache.clear()
    return translator.translate(text, language)


def recommend_threads(runs: list[dict]) -> tuple[int, dict]:
    """Rank reviewed runs; never silently discard an unusually slow sample."""
    totals: dict[int, list[float]] = {}
    for run in runs:
        if run.get("exclude_from_selection", False):
            continue
        totals.setdefault(run["threads"], []).append(run["total_seconds"])
    medians = {threads: statistics.median(values) for threads, values in totals.items()}
    if not medians:
        raise RuntimeError("没有完成可用于对照的线程配置")
    fastest = min(medians.values())
    selected = min(threads for threads, elapsed in medians.items() if elapsed <= fastest * 1.10)
    return selected, {str(k): round(v, 6) for k, v in sorted(medians.items())}


def _check_memory(sampler, minimum_mb: int, stage: str) -> dict:
    snapshot = sampler.sample_once()
    available = snapshot.get("ram_available_bytes")
    if available is not None and available < minimum_mb * 1024 ** 2:
        raise RuntimeError(f"{stage}可用内存低于 {minimum_mb} MiB，停止基准，避免继续加压")
    return snapshot


def benchmark(*, rounds: int = 2, minimum_before_mb: int = 1024,
              minimum_loaded_mb: int = 512) -> dict:
    if rounds < 2:
        raise ValueError("至少需要 2 轮对照")
    bootstrap()
    from .config import translation_path
    from .performance import ResourceSampler
    from .translation import LocalTranslator

    sampler = ResourceSampler()
    report = {
        "method": "CPU INT8 M2M100; beam=1; return_scores=False; one model at a time",
        "corpus": [{"language": language, "source": text} for language, text in CORPUS],
        "rounds_requested": rounds,
        "logical_cpu_count": os.cpu_count(),
        "selection_rule": "Within 10% of fastest median corpus time, prefer fewer threads",
        "memory_guard_mib": {"before_load": minimum_before_mb, "loaded": minimum_loaded_mb},
        "anomaly_rule": "A corpus taking >30s with average_busy_cores<0.5 requires manual review; it is not silently excluded. Explicit exclude_from_selection flags retain a reason and raw measurements.",
        "limitations": "Short authored DE/EN text corpus, current background load; sampled RSS is not a guaranteed peak. Not a 90-minute hardware soak or translation-quality evaluation.",
        "initial_resources": sampler.sample_once(),
        "runs": [],
    }
    try:
        for round_index in range(rounds):
            order = (4, 6, 8) if round_index % 2 == 0 else (8, 6, 4)
            for threads in order:
                _check_memory(sampler, minimum_before_mb, "加载前")
                translator = LocalTranslator(translation_path(), cpu_threads=threads, beam_size=1)
                try:
                    started = time.perf_counter()
                    translator.load()
                    loaded = time.perf_counter() - started
                    resources = _check_memory(sampler, minimum_loaded_mb, "加载后")
                    _uncached_translate(translator, "The lecture is starting now.", "en")
                    _check_memory(sampler, minimum_loaded_mb, "预热期间")
                    _uncached_translate(translator, "Die Vorlesung beginnt jetzt.", "de")
                    wall_start, cpu_start = time.perf_counter(), time.process_time()
                    results = []
                    for language, text in CORPUS:
                        _check_memory(sampler, minimum_loaded_mb, "运行期间")
                        tick = time.perf_counter()
                        translated = _uncached_translate(translator, text, language)
                        results.append(dict(language=language, seconds=time.perf_counter() - tick,
                                            translation=translated))
                    elapsed = time.perf_counter() - wall_start
                    cpu_seconds = time.process_time() - cpu_start
                    run = dict(round=round_index + 1, threads=threads, load_seconds=loaded,
                               total_seconds=elapsed, process_cpu_seconds=cpu_seconds,
                               average_busy_cores=cpu_seconds / elapsed,
                               machine_cpu_percent=100 * cpu_seconds / elapsed / (os.cpu_count() or 1),
                               loaded_resources=resources, after_resources=sampler.sample_once(),
                               sentences=results)
                    run["anomaly_review_required"] = elapsed > 30 and cpu_seconds / elapsed < .5
                    run["exclude_from_selection"] = False
                    report["runs"].append(run)
                    print(f"round={round_index + 1} threads={threads} corpus={elapsed:.3f}s "
                          f"CPU={run['machine_cpu_percent']:.1f}%", flush=True)
                finally:
                    translator.close()
                    del translator
                    gc.collect()
        if any(run["anomaly_review_required"] for run in report["runs"]):
            raise RuntimeError("发现长时间低 CPU 停顿，需要人工复核原始数据；未自动删除或排序该样本")
        chosen, medians = recommend_threads(report["runs"])
        report.update(recommended_threads=chosen, median_corpus_seconds=medians,
                      completed=True)
    except Exception as exc:
        report.update(completed=False, error=str(exc))
    finally:
        report["final_resources"] = sampler.sample_once()
        sampler.close()
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="本地翻译 CPU 4/6/8 线程对照，不加载 GPU 识别模型")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--min-before-mib", type=int, default=1024)
    parser.add_argument("--min-loaded-mib", type=int, default=512)
    parser.add_argument("--output", type=Path, default=Path("cpu-tuning.json"))
    args = parser.parse_args(argv)
    result = benchmark(rounds=args.rounds, minimum_before_mb=args.min_before_mib,
                       minimum_loaded_mb=args.min_loaded_mib)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告：{args.output.resolve()}", flush=True)
    if result.get("completed"):
        print(f"建议线程数：{result['recommended_threads']}", flush=True)
        return 0
    print(result.get("error", "基准未完成"), flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
