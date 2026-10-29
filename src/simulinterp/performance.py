"""Bounded, read-only resource sampling; no subprocess, CUDA context or model load.

Windows values come from GlobalMemoryStatusEx, GetProcessMemoryInfo,
GetSystemTimes and GetProcessTimes. GPU values are board-wide NVML readings,
not per-process allocations. Unsupported readings are None rather than zero.
"""
from __future__ import annotations

import ctypes
import logging
import os
from pathlib import Path
from threading import Event
import time
from typing import Callable

from PySide6.QtCore import QThread, Signal


log = logging.getLogger(__name__)
FIELDS = (
    "ram_total_bytes", "ram_available_bytes", "process_rss_bytes",
    "system_cpu_percent", "process_cpu_percent", "gpu_total_bytes",
    "gpu_used_bytes", "gpu_util_percent", "gpu_temperature_c",
)


class _MemoryStatus(ctypes.Structure):
    _fields_ = [("length", ctypes.c_uint32), ("load", ctypes.c_uint32)] + [
        (name, ctypes.c_uint64) for name in (
            "total_physical", "available_physical", "total_pagefile",
            "available_pagefile", "total_virtual", "available_virtual",
            "available_extended_virtual",
        )
    ]


class _ProcessMemory(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_uint32), ("page_faults", ctypes.c_uint32)] + [
        (name, ctypes.c_size_t) for name in (
            "peak_working_set", "working_set", "peak_paged_pool", "paged_pool",
            "peak_nonpaged_pool", "nonpaged_pool", "pagefile", "peak_pagefile",
        )
    ]


class _FileTime(ctypes.Structure):
    _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_uint32)]

    @property
    def ticks(self) -> int:
        return (self.high << 32) | self.low


class _CpuDeltas:
    """Keep only the preceding sample, never a growing performance history."""

    def __init__(self, cpu_count: int | None = None):
        self.cpu_count = max(1, cpu_count or os.cpu_count() or 1)
        self.system_previous = None
        self.process_previous = None

    def system(self, idle: int, kernel: int, user: int) -> float | None:
        previous = self.system_previous
        self.system_previous = (idle, kernel + user)
        if previous is None:
            return None
        delta = kernel + user - previous[1]
        idle_delta = idle - previous[0]
        if delta <= 0 or idle_delta < 0:
            return None
        # GetSystemTimes kernel ticks include the idle time.
        return max(0.0, min(100.0, 100.0 * (delta - idle_delta) / delta))

    def process(self, ticks: int, now: float) -> float | None:
        previous = self.process_previous
        self.process_previous = (ticks, now)
        if previous is None:
            return None
        elapsed = now - previous[1]
        delta = ticks - previous[0]
        if elapsed <= 0 or delta < 0:
            return None
        return max(0.0, min(100.0, delta / 10_000_000 / elapsed / self.cpu_count * 100))


class _WindowsMetrics:
    def __init__(self):
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        self._memory = kernel.GlobalMemoryStatusEx
        self._memory.argtypes = [ctypes.POINTER(_MemoryStatus)]
        self._memory.restype = ctypes.c_int
        current = kernel.GetCurrentProcess
        current.argtypes = []
        current.restype = ctypes.c_void_p
        self._process = current()  # Pseudo-handle; it must not be closed.
        self._rss = psapi.GetProcessMemoryInfo
        self._rss.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ProcessMemory), ctypes.c_uint32]
        self._rss.restype = ctypes.c_int
        self._system_time = kernel.GetSystemTimes
        self._system_time.argtypes = [ctypes.POINTER(_FileTime)] * 3
        self._system_time.restype = ctypes.c_int
        self._process_time = kernel.GetProcessTimes
        self._process_time.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(_FileTime)] * 4
        self._process_time.restype = ctypes.c_int
        self._cpu = _CpuDeltas()

    def sample(self) -> dict:
        values = {}
        memory = _MemoryStatus()
        memory.length = ctypes.sizeof(memory)
        if self._memory(ctypes.byref(memory)):
            values.update(ram_total_bytes=memory.total_physical,
                          ram_available_bytes=memory.available_physical)
        process_memory = _ProcessMemory()
        process_memory.cb = ctypes.sizeof(process_memory)
        if self._rss(self._process, ctypes.byref(process_memory), process_memory.cb):
            values["process_rss_bytes"] = process_memory.working_set
        idle, kernel, user = _FileTime(), _FileTime(), _FileTime()
        if self._system_time(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
            values["system_cpu_percent"] = self._cpu.system(idle.ticks, kernel.ticks, user.ticks)
        created, exited, kernel, user = (_FileTime() for _ in range(4))
        if self._process_time(self._process, ctypes.byref(created), ctypes.byref(exited),
                              ctypes.byref(kernel), ctypes.byref(user)):
            values["process_cpu_percent"] = self._cpu.process(kernel.ticks + user.ticks, time.monotonic())
        return values


class _NvmlMemory(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in ("total", "free", "used")]


class _NvmlUtilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint32), ("memory", ctypes.c_uint32)]


def _nvml_library():
    if os.name == "nt":
        # Avoid loading arbitrary DLLs from the current working directory/PATH.
        candidates = [Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvml.dll",
                      Path(os.environ.get("ProgramFiles", r"C:\Program Files")) /
                      "NVIDIA Corporation" / "NVSMI" / "nvml.dll"]
        for candidate in candidates:
            if candidate.is_file():
                try:
                    return ctypes.CDLL(str(candidate))
                except OSError:
                    pass
        raise OSError("NVIDIA NVML is unavailable")
    return ctypes.CDLL("libnvidia-ml.so.1")


class _NvmlMetrics:
    """Read NVIDIA board 0, matching this app's default CUDA device 0."""

    def __init__(self):
        self._library = _nvml_library()
        self._initialized = False
        init = getattr(self._library, "nvmlInit_v2", None) or self._library.nvmlInit
        init.argtypes, init.restype = [], ctypes.c_int
        shutdown = self._library.nvmlShutdown
        shutdown.argtypes, shutdown.restype = [], ctypes.c_int
        self._shutdown = shutdown
        if init() != 0:
            raise OSError("NVIDIA NVML initialization failed")
        self._initialized = True
        try:
            handle = (getattr(self._library, "nvmlDeviceGetHandleByIndex_v2", None)
                      or self._library.nvmlDeviceGetHandleByIndex)
            handle.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)]
            handle.restype = ctypes.c_int
            self._device = ctypes.c_void_p()
            if handle(0, ctypes.byref(self._device)) != 0:
                raise OSError("NVIDIA GPU 0 is unavailable")
            self._memory = self._function("nvmlDeviceGetMemoryInfo", [_NvmlMemory])
            self._utilization = self._function("nvmlDeviceGetUtilizationRates", [_NvmlUtilization])
            self._temperature = getattr(self._library, "nvmlDeviceGetTemperature", None)
            if self._temperature is not None:
                self._temperature.argtypes = [ctypes.c_void_p, ctypes.c_uint32,
                                              ctypes.POINTER(ctypes.c_uint32)]
                self._temperature.restype = ctypes.c_int
        except Exception:
            self.close()
            raise

    def _function(self, name: str, structures: list):
        function = getattr(self._library, name, None)
        if function is not None:
            function.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(s) for s in structures]
            function.restype = ctypes.c_int
        return function

    @staticmethod
    def _valid(result: int) -> bool:
        if result == 15:  # NVML_ERROR_GPU_IS_LOST: reinitialize on a later poll.
            raise OSError("NVIDIA GPU became inaccessible")
        return result == 0

    def sample(self) -> dict:
        values = {}
        memory = _NvmlMemory()
        if self._memory and self._valid(self._memory(self._device, ctypes.byref(memory))):
            values.update(gpu_total_bytes=memory.total, gpu_used_bytes=memory.used)
        utilization = _NvmlUtilization()
        if self._utilization and self._valid(self._utilization(self._device, ctypes.byref(utilization))):
            values["gpu_util_percent"] = int(utilization.gpu)
        temperature = ctypes.c_uint32()
        if self._temperature and self._valid(self._temperature(self._device, 0, ctypes.byref(temperature))):
            values["gpu_temperature_c"] = int(temperature.value)
        return values

    def close(self):
        if self._initialized:
            self._initialized = False
            self._shutdown()


class _UnavailableMetrics:
    def sample(self):
        return {}


class ResourceSampler:
    """Collect one snapshot, isolating unavailable providers and retrying in 30s."""

    def __init__(self, *, system_factory: Callable | None = None,
                 gpu_factory: Callable | None = None, clock: Callable = time.monotonic):
        self._clock = clock
        self._factories = {"system": system_factory or (_WindowsMetrics if os.name == "nt" else _UnavailableMetrics),
                           "gpu": gpu_factory or _NvmlMetrics}
        self._providers = {}
        self._retry_after = {}
        self._closed = False

    def sample_once(self) -> dict:
        values = dict.fromkeys(FIELDS)
        if self._closed:
            return values
        now = self._clock()
        for name, factory in self._factories.items():
            provider = self._providers.get(name)
            if provider is None and now >= self._retry_after.get(name, 0):
                try:
                    provider = factory()
                    self._providers[name] = provider
                except Exception:
                    self._retry_after[name] = now + 30
                    continue
            if provider is None:
                continue
            try:
                values.update(provider.sample())
            except Exception:
                self._release(name)
                self._retry_after[name] = now + 30
        return values

    def _release(self, name):
        provider = self._providers.pop(name, None)
        if provider is not None and hasattr(provider, "close"):
            try:
                provider.close()
            except Exception:
                log.debug("Resource monitor cleanup failed", exc_info=True)

    def close(self):
        self._closed = True
        for name in tuple(self._providers):
            self._release(name)


class ResourceMonitor(QThread):
    """Two-second monitor with interruptible waits and no GUI-thread polling."""

    snapshot = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stop_event = Event()

    def request_stop(self):
        self._stop_event.set()

    def run(self):
        sampler = ResourceSampler()
        try:
            while not self._stop_event.is_set():
                try:
                    self.snapshot.emit(sampler.sample_once())
                except Exception:
                    # Monitoring must never interrupt or terminate transcription.
                    log.debug("Resource monitor sample failed", exc_info=True)
                self._stop_event.wait(2.0)
        finally:
            sampler.close()
