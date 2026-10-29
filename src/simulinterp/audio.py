"""Windows WASAPI capture. Device access is lazy and only starts on request."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import logging
import math
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread, current_thread
import time
from typing import Callable, Literal

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AudioDevice:
    index: int
    name: str
    kind: Literal["microphone", "loopback"]
    sample_rate: int
    channels: int
    is_default: bool = False


def _audio_module():
    try:
        return importlib.import_module("pyaudiowpatch")
    except ImportError as exc:
        raise RuntimeError("未安装 Windows 音频组件，请先运行 setup.ps1。") from exc


def list_audio_devices() -> list[AudioDevice]:
    """Return WASAPI input devices and speaker loopback devices, without recording.

    PortAudio indices identify devices for the current enumeration only. Call this
    again after plugging in a device or changing the Windows output device.
    """
    pa = _audio_module()
    with pa.PyAudio() as audio:
        try:
            host = audio.get_host_api_info_by_type(pa.paWASAPI)
        except OSError as exc:
            raise RuntimeError("Windows WASAPI 不可用，请检查音频服务与声卡驱动。") from exc
        default_input = int(host.get("defaultInputDevice", -1))
        try:
            default_loopback = int(audio.get_default_wasapi_loopback()["index"])
        except (OSError, LookupError):
            default_loopback = -1
        result = []
        for index in range(audio.get_device_count()):
            info = audio.get_device_info_by_index(index)
            if int(info["hostApi"]) != int(host["index"]):
                continue
            channels = int(info.get("maxInputChannels", 0))
            if channels < 1:
                continue
            loopback = bool(info.get("isLoopbackDevice", False))
            result.append(AudioDevice(
                index=int(info["index"]),
                name=str(info["name"]),
                kind="loopback" if loopback else "microphone",
                sample_rate=int(info["defaultSampleRate"]),
                channels=channels,
                is_default=int(info["index"]) == (default_loopback if loopback else default_input),
            ))
        return sorted(result, key=lambda item: (item.kind != "loopback", not item.is_default, item.name))


class MonoResampler:
    """Band-limited streaming resampling with one block of lookahead.

    Each block is aligned to the rational sample-rate ratio. Previous and next
    blocks provide FIR context, avoiding filter discontinuities at callback edges.
    The added buffering delay is approximately 20 ms for 44.1/48 kHz devices.
    """

    def __init__(self, source_rate: int, target_rate: int = 16000):
        if source_rate <= 0 or target_rate <= 0:
            raise ValueError("Sample rates must be positive")
        self.source_rate = source_rate
        self.target_rate = target_rate
        divisor = math.gcd(source_rate, target_rate)
        self._up, self._down = target_rate // divisor, source_rate // divisor
        units = max(1, math.ceil(source_rate * 0.02 / self._down))
        self._block_in, self._block_out = units * self._down, units * self._up
        self._history = np.zeros(self._block_in, dtype=np.float32)
        self._pending = np.empty(0, dtype=np.float32)
        self._resample_poly = None

    def prepare(self) -> None:
        """Pay the cold DSP import/initialization cost before a device starts."""
        if self.source_rate == self.target_rate or self._resample_poly is not None:
            return
        from scipy.signal import resample_poly

        resample_poly(np.zeros(self._block_in * 3, dtype=np.float32), self._up, self._down)
        self._resample_poly = resample_poly

    def process(self, mono: np.ndarray) -> np.ndarray:
        samples = np.asarray(mono, dtype=np.float32)
        if samples.ndim != 1:
            raise ValueError("Resampler requires mono audio")
        if self.source_rate == self.target_rate:
            return samples.copy()
        self._pending = np.concatenate((self._pending, samples))
        output = []
        while self._pending.size >= self._block_in * 2:
            output.append(self._consume_block())
        return np.concatenate(output) if output else np.empty(0, dtype=np.float32)

    def _consume_block(self) -> np.ndarray:
        self.prepare()

        current = self._pending[:self._block_in]
        following = self._pending[self._block_in:self._block_in * 2]
        window = np.concatenate((self._history, current, following))
        resampled = self._resample_poly(window, self._up, self._down)
        result = resampled[self._block_out:self._block_out * 2].astype(np.float32, copy=True)
        self._history = current.copy()
        self._pending = self._pending[self._block_in:].copy()
        return result

    def flush(self) -> np.ndarray:
        if not self._pending.size:
            return np.empty(0, dtype=np.float32)
        expected = math.ceil(self._pending.size * self.target_rate / self.source_rate)
        blocks = math.ceil(self._pending.size / self._block_in)
        self._pending = np.pad(self._pending, (0, (blocks + 1) * self._block_in - self._pending.size))
        output = [self._consume_block() for _ in range(blocks)]
        self._pending = np.empty(0, dtype=np.float32)
        return np.concatenate(output)[:expected]


@dataclass(frozen=True)
class _AudioPacket:
    data: bytes
    start_frame: int
    frame_count: int


class AudioCapture:
    """Capture on a worker thread and deliver mono float32 audio at 16 kHz.

    ``on_audio`` and ``on_error`` run on the worker thread. The receiver should
    enqueue audio quickly; neural inference must not run inside these callbacks.
    """

    def __init__(
        self,
        device_index: int,
        on_audio: Callable[[np.ndarray], None],
        on_error: Callable[[str], None],
        *,
        on_warning: Callable[[str], None] | None = None,
        on_ready: Callable[[], None] | None = None,
    ):
        self.device_index = device_index
        self.on_audio = on_audio
        self.on_error = on_error
        self.on_warning = on_warning
        self.on_ready = on_ready
        self._stop = Event()
        self._thread: Thread | None = None
        self._lock = Lock()
        self.driver_overflows = 0
        self.queue_dropped_frames = 0
        self.padded_frames = 0
        self._sample_rate = 0
        self._last_warning = float('-inf')
        self._reported_recoveries = (0, 0)

    def _notify_recovery(self, rate: int, *, force: bool = False) -> None:
        counts = (self.driver_overflows, self.queue_dropped_frames)
        if counts == self._reported_recoveries:
            return
        now = time.monotonic()
        if not force and now - self._last_warning < 3.0:
            return
        details = []
        if self.driver_overflows:
            details.append(f"设备短暂丢帧 {self.driver_overflows} 次")
        if self.queue_dropped_frames:
            details.append(f"跳过积压音频 {self.queue_dropped_frames / rate:.2f} 秒")
        message = "音频采集已自动恢复 · " + "，".join(details)
        self._last_warning = now
        self._reported_recoveries = counts
        logger.warning(message)
        if self.on_warning is not None:
            try:
                self.on_warning(message)
            except Exception:
                logger.exception("Audio warning observer failed")

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def statistics(self) -> dict:
        """Constant-size counters; never copy the audio buffer for monitoring."""
        rate = self._sample_rate
        return {
            "driver_overflows": self.driver_overflows,
            "queue_dropped_seconds": self.queue_dropped_frames / rate if rate else 0.0,
        }

    def start(self) -> None:
        with self._lock:
            if self.is_running:
                raise RuntimeError("音频采集已启动。")
            self._stop.clear()
            self.driver_overflows = self.queue_dropped_frames = self.padded_frames = 0
            self._last_warning = float('-inf')
            self._reported_recoveries = (0, 0)
            self._thread = Thread(target=self._run, name="wasapi-capture", daemon=True)
            self._thread.start()

    def request_stop(self) -> None:
        """Stop accepting device packets without blocking the calling UI thread."""
        self._stop.set()

    def stop(self, timeout: float = 3.0) -> None:
        self.request_stop()
        worker = self._thread
        if worker is not None and worker is not current_thread():
            worker.join(timeout)
            if worker.is_alive():
                raise RuntimeError("音频驱动未及时停止，请关闭应用后检查音频设备。")

    def _run(self) -> None:
        stream = None
        audio = None
        try:
            pa = _audio_module()
            audio = pa.PyAudio()
            device = audio.get_device_info_by_index(self.device_index)
            channels = int(device.get("maxInputChannels", 0))
            if channels < 1:
                raise RuntimeError("所选设备不支持录音，请刷新设备列表。")
            rate = int(device["defaultSampleRate"])
            self._sample_rate = rate
            frames_per_buffer = max(1, round(rate * 0.02))
            # Five seconds is a finite safety margin, not a fixed playback delay.
            packets: Queue[_AudioPacket] = Queue(maxsize=250)
            resampler = MonoResampler(rate)
            resampler.prepare()
            native_cursor = 0
            previous_adc = None
            previous_frames = 0

            def callback(in_data, frame_count, time_info, status_flags):
                nonlocal native_cursor, previous_adc, previous_frames
                if self._stop.is_set():
                    return None, pa.paComplete
                adc = time_info.get("input_buffer_adc_time", 0.0)
                adc = float(adc) if isinstance(adc, (float, int)) else 0.0
                if not math.isfinite(adc) or adc <= 0:
                    adc = None
                if status_flags & getattr(pa, "paInputOverflow", 2):
                    self.driver_overflows += 1
                    # The flag alone gives no lost-frame count. Only a valid
                    # ADC clock can supply a gap; do not invent one otherwise.
                    if adc is not None and previous_adc is not None:
                        gap = round((adc - previous_adc) * rate) - previous_frames
                        if round(rate * .002) <= gap <= rate * 60:
                            native_cursor += gap
                previous_adc, previous_frames = adc, frame_count
                packet = _AudioPacket(in_data or b"", native_cursor, frame_count)
                native_cursor += frame_count
                try:
                    packets.put_nowait(packet)
                except Full:
                    # Keep current audio, rather than aborting a healthy device.
                    # Frame offsets let the consumer preserve the skipped time.
                    try:
                        dropped = packets.get_nowait()
                        self.queue_dropped_frames += dropped.frame_count
                    except Empty:
                        pass
                    packets.put_nowait(packet)
                return None, pa.paContinue

            stream = audio.open(
                format=pa.paInt16,
                channels=channels,
                rate=rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=frames_per_buffer,
                stream_callback=callback,
                start=False,
            )
            if self._stop.is_set():
                return
            stream.start_stream()
            if self.on_ready is not None and not self._stop.is_set():
                self.on_ready()
            expected_frame = 0

            def deliver(mono):
                converted = resampler.process(mono)
                if converted.size:
                    self.on_audio(converted)

            def fill_gap(frames):
                # Bound temporary allocations even if a device's clock jumps.
                self.padded_frames += frames
                while frames > 0:
                    length = min(frames, rate)
                    deliver(np.zeros(length, dtype=np.float32))
                    frames -= length
            # Keep packets already accepted before Stop so the last spoken word
            # survives. The callback stops adding packets once _stop is set.
            while not self._stop.is_set() or not packets.empty():
                try:
                    packet = packets.get(timeout=0.1)
                except Empty:
                    if self._stop.is_set():
                        break
                    if not stream.is_active():
                        raise RuntimeError("音频设备已停止，请检查设备连接并刷新设备列表。")
                    continue
                self._notify_recovery(rate)
                if packet.start_frame > expected_frame:
                    fill_gap(packet.start_frame - expected_frame)
                interleaved = np.frombuffer(packet.data, dtype=np.int16)
                if interleaved.size and interleaved.size != packet.frame_count * channels:
                    raise RuntimeError("音频设备返回了不完整的数据帧。")
                if interleaved.size:
                    mono = interleaved.reshape(-1, channels).mean(axis=1, dtype=np.float32) / 32768.0
                    deliver(mono)
                else:
                    fill_gap(packet.frame_count)
                expected_frame = packet.start_frame + packet.frame_count
            if native_cursor > expected_frame:
                fill_gap(native_cursor - expected_frame)
            tail = resampler.flush()
            if tail.size:
                self.on_audio(tail)
            self._notify_recovery(rate, force=True)
        except Exception as exc:
            if not self._stop.is_set():
                self._stop.set()
                self.on_error(f"音频采集失败：{exc}")
        finally:
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
            if audio is not None:
                audio.terminate()
