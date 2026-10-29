from types import SimpleNamespace
import pytest

from simulinterp.benchmark_translation import _check_memory, recommend_threads


def test_tuning_prefers_fewer_threads_when_latency_is_within_ten_percent():
    runs = [dict(threads=t, total_seconds=s) for t, s in
            [(4, 1.05), (6, 1.0), (8, 1.03), (8, 1.01), (6, 1.0), (4, 1.07)]]
    chosen, medians = recommend_threads(runs)
    assert chosen == 4
    assert medians == {"4": 1.06, "6": 1.0, "8": 1.02}


def test_tuning_keeps_more_threads_when_they_are_materially_faster():
    runs = [dict(threads=t, total_seconds=s) for t, s in
            [(4, 1.6), (6, 1.4), (8, 1.0), (8, 1.1), (6, 1.5), (4, 1.7)]]
    assert recommend_threads(runs)[0] == 8


def test_audited_excluded_run_does_not_corrupt_latency_ranking():
    runs = [dict(threads=4, total_seconds=2.8), dict(threads=6, total_seconds=2.3),
            dict(threads=8, total_seconds=2.0),
            dict(threads=4, total_seconds=641.9, exclude_from_selection=True,
                 exclusion_reason="Manual review: long low-CPU stall; cause unknown")]
    choice, medians = recommend_threads(runs)
    assert choice == 8
    assert medians["4"] == 2.8
    # A slow sample is included unless someone explicitly flags it for exclusion.
    runs[-1]["exclude_from_selection"] = False
    assert recommend_threads(runs)[1]["4"] > 300


def test_memory_guard_stops_pressure_and_does_not_fake_unknown_readings():
    sampler = SimpleNamespace(sample_once=lambda: {"ram_available_bytes": 400 * 1024 ** 2})
    with pytest.raises(RuntimeError, match="512"):
        _check_memory(sampler, 512, "运行期间")
    sampler.sample_once = lambda: {"ram_available_bytes": None}
    assert _check_memory(sampler, 512, "运行期间")["ram_available_bytes"] is None
