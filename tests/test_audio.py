from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.signal import resample_poly

from simulinterp.audio import AudioCapture, MonoResampler, list_audio_devices


class ResamplerTests(unittest.TestCase):
    def test_44100_and_48000_streams_match_whole_file_resampling(self):
        for rate in (44100, 48000):
            with self.subTest(rate=rate):
                samples = (0.4 * np.sin(2 * np.pi * 700 * np.arange(rate + 73) / rate)).astype(np.float32)
                resampler = MonoResampler(rate)
                parts = [resampler.process(samples[offset:offset + 379]) for offset in range(0, samples.size, 379)]
                parts.append(resampler.flush())
                actual = np.concatenate(parts)
                expected = resample_poly(samples, 16000 // np.gcd(rate, 16000), rate // np.gcd(rate, 16000))
                self.assertEqual(actual.size, expected.size)
                np.testing.assert_allclose(actual, expected, atol=2e-6)
                self.assertEqual(resampler.flush().size, 0)

    def test_resampling_filters_frequencies_above_output_nyquist(self):
        rate = 48000
        samples = np.sin(2 * np.pi * 12000 * np.arange(rate) / rate).astype(np.float32)
        resampler = MonoResampler(rate)
        actual = np.concatenate((resampler.process(samples), resampler.flush()))
        self.assertLess(float(np.sqrt(np.mean(actual[100:-100] ** 2))), 0.002)

    def test_native_16k_is_copied(self):
        samples = np.arange(321, dtype=np.float32)
        actual = MonoResampler(16000).process(samples)
        np.testing.assert_array_equal(actual, samples)
        self.assertIsNot(actual, samples)


class FakeAudio:
    devices = [
        dict(index=0, name="MME microphone", hostApi=0, maxInputChannels=1, defaultSampleRate=44100),
        dict(index=1, name="Microphone", hostApi=1, maxInputChannels=2, defaultSampleRate=48000),
        dict(index=2, name="Speakers", hostApi=1, maxInputChannels=0, defaultSampleRate=48000),
        dict(index=3, name="Speakers [Loopback]", hostApi=1, maxInputChannels=2, defaultSampleRate=48000, isLoopbackDevice=True),
    ]

    def __init__(self):
        self.terminated = False
        self.stream = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.terminate()

    def get_host_api_info_by_type(self, host_type):
        return dict(index=1, defaultInputDevice=1, defaultOutputDevice=2)

    def get_default_wasapi_loopback(self):
        return self.devices[3]

    def get_device_count(self):
        return len(self.devices)

    def get_device_info_by_index(self, index):
        return self.devices[index]

    def terminate(self):
        self.terminated = True

    def open(self, **kwargs):
        self.open_kwargs = kwargs
        self.stream = FakeStream(kwargs["stream_callback"])
        return self.stream


class FakeStream:
    def __init__(self, callback):
        self.callback = callback
        self.closed = False

    def start_stream(self):
        # Simulate a device callback; no hardware is opened in these tests.
        stereo = np.tile(np.array([16384, 0], dtype=np.int16), 960)
        for _ in range(3):
            self.callback(stereo.tobytes(), 960, {}, 0)

    def is_active(self):
        return not self.closed

    def close(self):
        self.closed = True


class CaptureTests(unittest.TestCase):
    def fake_module(self, audio):
        return SimpleNamespace(PyAudio=lambda: audio, paWASAPI=13, paInt16=8, paContinue=0, paComplete=1, paAbort=2, paInputOverflow=2)

    def test_enumeration_returns_wasapi_inputs_and_default_loopback(self):
        audio = FakeAudio()
        with patch("simulinterp.audio._audio_module", return_value=self.fake_module(audio)):
            devices = list_audio_devices()
        self.assertEqual([device.index for device in devices], [3, 1])
        self.assertEqual([device.kind for device in devices], ["loopback", "microphone"])
        self.assertTrue(all(device.is_default for device in devices))
        self.assertTrue(audio.terminated)

    def test_capture_downmixes_and_cleans_up(self):
        audio = FakeAudio()
        received = []
        errors = []
        done = Event()

        def on_audio(samples):
            received.append(samples)
            capture.stop()
            done.set()

        capture = AudioCapture(3, on_audio, errors.append)
        with patch("simulinterp.audio._audio_module", return_value=self.fake_module(audio)):
            capture.start()
            self.assertTrue(done.wait(3))
            capture.stop()
        self.assertFalse(errors)
        self.assertTrue(audio.terminated)
        self.assertTrue(audio.stream.closed)
        self.assertEqual(received[0].dtype, np.float32)
        self.assertAlmostEqual(float(np.mean(received[0][50:])), 0.25, places=3)
        # Stop after the first callback still preserves all three packets that
        # the fake device had accepted, including the resampler's final tail.
        self.assertEqual(sum(part.size for part in received), 960)

    def test_stop_during_empty_poll_flushes_resampler_tail(self):
        # WASAPI can become inactive while the worker waits for a packet.
        # An ordinary Stop must still flush the resampler's retained FIR tail.
        from queue import Queue, Empty

        audio = FakeAudio()
        received, errors = [], []
        capture = AudioCapture(3, received.append, errors.append)

        class StopOnEmpty(Queue):
            def get(self, *args, **kwargs):
                if self.empty():
                    capture.request_stop()
                    audio.stream.closed = True
                    raise Empty
                return super().get(*args, **kwargs)

        with patch("simulinterp.audio._audio_module", return_value=self.fake_module(audio)), patch("simulinterp.audio.Queue", StopOnEmpty):
            capture.start()
            capture._thread.join(3)
        self.assertFalse(capture.is_running)
        self.assertFalse(errors)
        self.assertEqual(sum(part.size for part in received), 960)

    def test_capture_reports_initialization_error(self):
        done = Event()
        errors = []

        def on_error(message):
            errors.append(message)
            done.set()

        capture = AudioCapture(0, lambda samples: None, on_error)
        with patch("simulinterp.audio._audio_module", side_effect=RuntimeError("missing audio component")):
            capture.start()
            self.assertTrue(done.wait(3))
            capture.stop()
        self.assertIn("missing audio component", errors[0])
        self.assertFalse(capture.is_running)


if __name__ == "__main__":
    unittest.main()
