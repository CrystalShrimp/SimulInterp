"""Overflow regressions use deterministic fake WASAPI callbacks, never hardware."""
from __future__ import annotations

import builtins
from queue import Empty, Queue
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

import simulinterp.audio as audio_module
from simulinterp.audio import AudioCapture, MonoResampler


def packet(value: int, *, frames: int = 320, flags: int = 0, adc_time=None):
    info = {} if adc_time is None else {"input_buffer_adc_time": adc_time}
    return (np.full(frames, value, dtype=np.int16).tobytes(), frames, info, flags)


def run_capture(program, *, rate=16000, queue_capacity=None, resampler_type=None,
                forbid_late_dsp_import=False, before_open=None, stop_in_start=False):
    """Produce callbacks synchronously, then stop only after the queue drains.

    Queue injection makes saturation deterministic instead of depending on a
    machine's thread scheduling or load. Empty get stops the fake device, so
    tests need neither sleeps nor an actual PortAudio callback thread.
    """
    parts, errors, warnings, ready, events, returns = [], [], [], [], [], []
    capture = None
    stream = None

    class StopWhenDrained(Queue):
        def get(self, *args, **kwargs):
            if self.empty():
                capture.request_stop()
                raise Empty
            return super().get(block=False)

    def make_queue(maxsize=0):
        return StopWhenDrained(maxsize=maxsize if queue_capacity is None else queue_capacity)

    class Stream:
        def __init__(self, callback):
            self.callback = callback
            self.started = False
            self.in_native_callback = False
            self.closed = False

        def start_stream(self):
            self.started = True
            events.append("native.start")
            for args in program:
                self.in_native_callback = True
                try:
                    returns.append(self.callback(*args))
                finally:
                    self.in_native_callback = False
            # Warnings must be delivered by the consumer, not inside a native
            # callback, where UI work or logging can create another overflow.
            assert not warnings
            if stop_in_start:
                capture.request_stop()
            events.append("native.start.returned")

        def is_active(self):
            return not self.closed

        def close(self):
            self.closed = True

    class Audio:
        terminated = False

        def get_device_info_by_index(self, index):
            return {"maxInputChannels": 1, "defaultSampleRate": rate}

        def open(self, **options):
            nonlocal stream
            if before_open:
                before_open()
            events.append("native.open")
            assert options["start"] is False
            stream = Stream(options["stream_callback"])
            return stream

        def terminate(self):
            self.terminated = True

    native = Audio()
    pa = SimpleNamespace(PyAudio=lambda: native, paInt16=8, paContinue=0,
                         paComplete=1, paAbort=2, paInputOverflow=2)

    def on_warning(message):
        assert not stream.in_native_callback
        warnings.append(message)
        events.append("warning")

    def on_ready():
        assert stream.started
        assert events[-1] == "native.start.returned"
        ready.append(True)
        events.append("ready")

    actual_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if forbid_late_dsp_import and name == "scipy.signal":
            assert stream is None or not stream.started, "SciPy was first resolved after native capture started"
        return actual_import(name, *args, **kwargs)

    with patch.object(audio_module, "_audio_module", return_value=pa), \
            patch.object(audio_module, "Queue", side_effect=make_queue), \
            patch("builtins.__import__", side_effect=guarded_import):
        capture = AudioCapture(0, parts.append, errors.append,
                               on_warning=on_warning, on_ready=on_ready)
        if resampler_type is None:
            capture._run()
        else:
            with patch.object(audio_module, "MonoResampler", resampler_type):
                capture._run()
    assert native.terminated
    assert stream is not None and stream.closed
    assert not errors, errors
    assert ready == ([] if stop_in_start else [True])
    assert all(result == (None, pa.paContinue) for result in returns), returns
    samples = np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)
    return SimpleNamespace(samples=samples, warnings=warnings, events=events, capture=capture)


def test_driver_overflow_warns_and_continues_using_valid_callback_audio():
    result = run_capture([packet(1000), packet(2000, flags=2), packet(3000)])
    assert result.samples.size == 960
    np.testing.assert_allclose(result.samples, np.repeat([1000, 2000, 3000], 320) / 32768)
    assert len(result.warnings) == 1
    assert result.capture.driver_overflows == 1
    assert result.capture.padded_frames == 0
    assert result.events.index("warning") > result.events.index("ready")


def test_queue_overflow_keeps_latest_audio_and_zero_fills_exact_dropped_time():
    result = run_capture([packet(value * 1000) for value in range(1, 7)], queue_capacity=2)
    # Four discarded 20 ms packets still occupy their original 80 ms on the
    # subtitle timeline; silently concatenating retained packets would lose it.
    assert result.samples.size == 6 * 320
    np.testing.assert_array_equal(result.samples[:4 * 320], 0)
    np.testing.assert_allclose(result.samples[4 * 320:], np.repeat([5000, 6000], 320) / 32768)
    assert len(result.warnings) == 1
    assert result.capture.queue_dropped_frames == 1280
    assert result.capture.padded_frames == 1280


def test_reported_overflow_uses_valid_adc_gap_without_shifting_later_audio():
    result = run_capture([
        packet(1000, adc_time=100.0),
        packet(2000, flags=2, adc_time=100.06),
    ])
    # First packet ends at 100.02; second starts at 100.06: precisely 40 ms lost.
    assert result.samples.size == 1280
    np.testing.assert_allclose(result.samples[:320], 1000 / 32768)
    np.testing.assert_array_equal(result.samples[320:960], 0)
    np.testing.assert_allclose(result.samples[960:], 2000 / 32768)
    assert len(result.warnings) == 1
    assert result.capture.padded_frames == 640


@pytest.mark.parametrize("adc_time", [None, 0.0, -1.0, 99.0, float("nan"), float("inf")])
def test_unusable_driver_timestamps_do_not_invent_a_missing_interval(adc_time):
    result = run_capture([packet(1000, adc_time=100.0), packet(2000, flags=2, adc_time=adc_time)])
    assert result.samples.size == 640
    np.testing.assert_allclose(result.samples, np.repeat([1000, 2000], 320) / 32768)
    assert len(result.warnings) == 1


def test_sub_two_millisecond_clock_jitter_is_not_treated_as_missing_audio():
    result = run_capture([
        packet(1000, adc_time=100.0),
        packet(2000, flags=2, adc_time=100.021),
    ])
    assert result.samples.size == 640
    assert len(result.warnings) == 1


def test_adc_clock_change_without_overflow_does_not_stretch_the_stream():
    result = run_capture([packet(1000, adc_time=100.0), packet(2000, adc_time=100.06)])
    assert result.samples.size == 640
    assert not result.warnings


def test_stop_arriving_during_native_start_does_not_announce_listening():
    result = run_capture([packet(1000), packet(2000)], stop_in_start=True)
    # Already accepted packets still drain, while the readiness notification is
    # withheld so the GUI cannot show Listening after the user requested Stop.
    assert result.samples.size == 640
    assert "ready" not in result.events
    assert not result.warnings


def test_repeated_driver_notifications_are_coalesced_instead_of_spamming_ui():
    result = run_capture([packet(1000, flags=2) for _ in range(12)])
    assert result.samples.size == 12 * 320
    assert len(result.warnings) == 1


def test_capture_prepares_dsp_before_opening_native_audio_and_never_imports_it_late():
    preparation = []

    class TracedResampler(MonoResampler):
        def prepare(self):
            preparation.append("prepared")
            return super().prepare()

    # Resolution must happen before open(), not on the first nonempty audio
    # block. The import guard also catches the former delayed scipy.signal import.
    def prepared_before_open():
        assert preparation, "Native audio opened before DSP preparation"

    result = run_capture([
        packet(1000, frames=960), packet(2000, frames=960), packet(3000, frames=960),
    ], rate=48000, resampler_type=TracedResampler,
        forbid_late_dsp_import=True, before_open=prepared_before_open)
    assert preparation
    assert result.samples.size == 960
    assert not result.warnings
