"""Resource sampling is independent of a GPU, native audio and model loading."""
import unittest
from unittest.mock import patch

from simulinterp.performance import FIELDS, ResourceMonitor, ResourceSampler, _CpuDeltas


class FakeProvider:
    def __init__(self, values=None):
        self.values = values or {}
        self.samples = 0
        self.closed = 0
        self.fail = False

    def sample(self):
        self.samples += 1
        if self.fail:
            raise OSError("device disappeared")
        return self.values.copy()

    def close(self):
        self.closed += 1


class ResourceSamplerTests(unittest.TestCase):
    def test_missing_gpu_still_reports_system_memory_and_unknown_gpu(self):
        system = FakeProvider({"ram_total_bytes": 16000, "ram_available_bytes": 8000})
        def unavailable():
            raise OSError("no driver")
        sampler = ResourceSampler(system_factory=lambda: system, gpu_factory=unavailable)
        sample = sampler.sample_once()
        self.assertEqual(set(sample), set(FIELDS))
        self.assertEqual(sample["ram_available_bytes"], 8000)
        self.assertIsNone(sample["gpu_used_bytes"])
        self.assertIsNone(sample["system_cpu_percent"])
        sampler.close()

    def test_provider_failure_clears_stale_values_and_retries_after_30_seconds(self):
        now = [0.0]
        system = FakeProvider({"ram_available_bytes": 8000})
        old_gpu = FakeProvider({"gpu_used_bytes": 2000})
        new_gpu = FakeProvider({"gpu_used_bytes": 1000})
        attempts = []
        def make_gpu():
            attempts.append(now[0])
            return old_gpu if len(attempts) == 1 else new_gpu
        sampler = ResourceSampler(system_factory=lambda: system, gpu_factory=make_gpu, clock=lambda: now[0])
        self.assertEqual(sampler.sample_once()["gpu_used_bytes"], 2000)
        old_gpu.fail = True
        now[0] = 2
        sample = sampler.sample_once()
        self.assertIsNone(sample["gpu_used_bytes"])
        self.assertEqual(sample["ram_available_bytes"], 8000)
        self.assertEqual(old_gpu.closed, 1)
        now[0] = 31.9
        self.assertIsNone(sampler.sample_once()["gpu_used_bytes"])
        self.assertEqual(len(attempts), 1)
        now[0] = 32
        self.assertEqual(sampler.sample_once()["gpu_used_bytes"], 1000)
        self.assertEqual(attempts, [0, 32])
        sampler.close()

    def test_initialization_failure_is_rate_limited_and_can_recover(self):
        now = [0.0]
        attempts = []
        provider = FakeProvider({"gpu_temperature_c": 55})
        def make_gpu():
            attempts.append(now[0])
            if len(attempts) == 1:
                raise OSError("driver asleep")
            return provider
        sampler = ResourceSampler(system_factory=FakeProvider, gpu_factory=make_gpu, clock=lambda: now[0])
        sampler.sample_once()
        for seconds in (2, 4, 28):
            now[0] = seconds
            self.assertIsNone(sampler.sample_once()["gpu_temperature_c"])
        now[0] = 30
        self.assertEqual(sampler.sample_once()["gpu_temperature_c"], 55)
        self.assertEqual(attempts, [0, 30])
        sampler.close()

    def test_close_is_idempotent_and_closed_sampler_does_not_restart(self):
        system, gpu = FakeProvider(), FakeProvider()
        sampler = ResourceSampler(system_factory=lambda: system, gpu_factory=lambda: gpu)
        sampler.sample_once()
        sampler.close()
        sampler.close()
        self.assertEqual(system.closed, 1)
        self.assertEqual(gpu.closed, 1)
        self.assertTrue(all(value is None for value in sampler.sample_once().values()))
        self.assertEqual(system.samples, 1)

    def test_ninety_minute_sampling_keeps_only_two_providers(self):
        now = [0.0]
        system = FakeProvider({"process_rss_bytes": 200_000_000})
        gpu = FakeProvider({"gpu_used_bytes": 1_000_000_000})
        sampler = ResourceSampler(system_factory=lambda: system, gpu_factory=lambda: gpu, clock=lambda: now[0])
        for index in range(2701):
            now[0] = index * 2.0
            sample = sampler.sample_once()
            self.assertEqual(len(sample), len(FIELDS))
            self.assertEqual(sample["process_rss_bytes"], 200_000_000)
        self.assertEqual(system.samples, 2701)
        self.assertEqual(gpu.samples, 2701)
        self.assertEqual(len(sampler._providers), 2)
        self.assertEqual(len(sampler._retry_after), 0)
        sampler.close()

    def test_cleanup_error_does_not_escape(self):
        class BrokenClose(FakeProvider):
            def close(self):
                raise OSError("driver gone")
        sampler = ResourceSampler(system_factory=FakeProvider, gpu_factory=BrokenClose)
        sampler.sample_once()
        sampler.close()


class CpuDeltaTests(unittest.TestCase):
    def test_system_kernel_includes_idle(self):
        cpu = _CpuDeltas(cpu_count=32)
        self.assertIsNone(cpu.system(100, 150, 50))
        # Delta total=100, delta idle=25, so 75% of the machine was busy.
        self.assertAlmostEqual(cpu.system(125, 200, 100), 75.0)

    def test_process_is_normalized_to_whole_machine(self):
        cpu = _CpuDeltas(cpu_count=32)
        self.assertIsNone(cpu.process(0, 10.0))
        # 8 CPU-seconds over 2 wall seconds = 4 of 32 logical cores = 12.5%.
        self.assertAlmostEqual(cpu.process(80_000_000, 12.0), 12.5)

    def test_zero_intervals_or_counter_reset_return_unknown(self):
        cpu = _CpuDeltas(cpu_count=1)
        cpu.system(100, 150, 50)
        self.assertIsNone(cpu.system(100, 150, 50))
        self.assertIsNone(cpu.system(10, 15, 5))
        cpu.process(100, 10)
        self.assertIsNone(cpu.process(100, 10))
        self.assertIsNone(cpu.process(50, 11))

    def test_usage_is_bounded_even_if_a_counter_is_inconsistent(self):
        cpu = _CpuDeltas(cpu_count=1)
        cpu.system(0, 0, 0)
        self.assertEqual(cpu.system(500, 100, 0), 0)
        cpu.process(0, 1)
        self.assertEqual(cpu.process(1_000_000_000, 2), 100)


class MonitorTests(unittest.TestCase):
    def test_monitor_stops_its_wait_and_releases_sampler(self):
        monitor = ResourceMonitor()
        sampler = FakeProvider()
        sampler.sample_once = lambda: {"ram_total_bytes": 1000}
        def receive(sample):
            self.assertEqual(sample["ram_total_bytes"], 1000)
            monitor.request_stop()
        monitor.snapshot.connect(receive)
        with patch("simulinterp.performance.ResourceSampler", return_value=sampler):
            monitor.run()
        self.assertEqual(sampler.closed, 1)

    def test_stop_before_thread_start_exits_without_sampling(self):
        monitor = ResourceMonitor()
        monitor.request_stop()
        sampler = FakeProvider()
        sampler.sample_once = lambda: self.fail("sampled after stop")
        with patch("simulinterp.performance.ResourceSampler", return_value=sampler):
            monitor.run()
        self.assertEqual(sampler.closed, 1)


if __name__ == "__main__":
    unittest.main()
