import unittest

import numpy as np

from simulinterp.segmenter import SpeechSegmenter


class LevelVad:
    def is_speech(self, pcm, sample_rate):
        return np.max(np.abs(np.frombuffer(pcm, dtype="<i2"))) > 1000


def tone(seconds):
    return np.full(round(seconds * 16000), 0.2, dtype=np.float32)


def silence(seconds):
    return np.zeros(round(seconds * 16000), dtype=np.float32)


class SegmenterTests(unittest.TestCase):
    def make(self, **kwargs):
        return SpeechSegmenter(vad=LevelVad(), **kwargs)

    def test_silence_never_creates_subtitles(self):
        segmenter = self.make()
        self.assertEqual(segmenter.feed(silence(40)), [])
        self.assertEqual(segmenter.flush(), [])

    def test_phrase_ends_after_silence_with_absolute_timestamp(self):
        segmenter = self.make()
        self.assertEqual(segmenter.feed(silence(2)), [])
        self.assertEqual(segmenter.feed(tone(1)), [])
        chunks = segmenter.feed(silence(0.6))
        self.assertEqual(len(chunks), 1)
        self.assertAlmostEqual(chunks[0].start, 1.8)
        self.assertAlmostEqual(chunks[0].end, 3.12)
        self.assertAlmostEqual(chunks[0].available_at, 3.5)
        self.assertEqual(len(chunks[0].samples), round(chunks[0].duration * 16000))
        self.assertEqual(segmenter.flush(), [])

    def test_continuous_speech_is_bounded_without_overlap_or_loss(self):
        segmenter = self.make(max_seconds=1)
        chunks = segmenter.feed(tone(3.25)) + segmenter.flush()
        self.assertEqual(len(chunks), 4)
        self.assertTrue(all(chunk.duration <= 1.000001 for chunk in chunks))
        self.assertTrue(all(chunk.available_at == chunk.end for chunk in chunks))
        for previous, following in zip(chunks, chunks[1:]):
            self.assertAlmostEqual(previous.end, following.start)
        self.assertEqual(sum(chunk.samples.size for chunk in chunks), 52000)
        self.assertAlmostEqual(chunks[-1].end, 3.25)

    def test_arbitrary_callback_sizes_produce_same_segmentation(self):
        audio = np.concatenate((silence(0.5), tone(1), silence(0.7), tone(0.8)))
        one = self.make()
        expected = one.feed(audio) + one.flush()
        split = self.make()
        actual = []
        for offset in range(0, len(audio), 137):
            actual.extend(split.feed(audio[offset:offset + 137]))
        actual.extend(split.flush())
        self.assertEqual([(x.start, x.end) for x in actual], [(x.start, x.end) for x in expected])
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a.samples, b.samples)

    def test_short_click_is_discarded(self):
        segmenter = self.make()
        self.assertEqual(segmenter.feed(np.concatenate((tone(0.04), silence(1)))), [])
        self.assertEqual(segmenter.flush(), [])

    def test_stop_flushes_partial_frame_and_is_idempotent(self):
        segmenter = self.make()
        segmenter.feed(tone(0.351))
        chunks = segmenter.flush()
        self.assertEqual(len(chunks), 1)
        self.assertAlmostEqual(chunks[0].end, 0.351)
        self.assertEqual(chunks[0].samples.size, 5616)
        self.assertEqual(segmenter.flush(), [])

    def test_invalid_audio_rejected(self):
        segmenter = self.make()
        with self.assertRaises(ValueError):
            segmenter.feed(np.zeros((2, 10)))
        with self.assertRaises(ValueError):
            segmenter.feed(np.array([float("nan")]))

    def test_large_pre_roll_still_respects_chunk_limit(self):
        segmenter = self.make(max_seconds=0.5, pre_roll_seconds=0.49, min_speech_seconds=0.02)
        chunks = segmenter.feed(np.concatenate((silence(1), tone(1)))) + segmenter.flush()
        self.assertTrue(chunks)
        self.assertTrue(all(chunk.duration <= 0.500001 for chunk in chunks))

    def adaptive(self, **options):
        return self.make(max_seconds=8, silence_seconds=.8,
                         soft_min_seconds=4, soft_silence_seconds=.3, **options)

    def test_adaptive_short_phrase_still_waits_for_normal_endpoint(self):
        segmenter = self.adaptive()
        self.assertEqual(segmenter.feed(tone(1)), [])
        self.assertEqual(segmenter.feed(silence(.78)), [])
        chunks = segmenter.feed(silence(.02))
        self.assertEqual(len(chunks), 1)
        self.assertAlmostEqual(chunks[0].available_at, 1.8)
        self.assertAlmostEqual(chunks[0].end, 1.12)

    def test_adaptive_must_not_count_endpoint_silence_as_retained_context(self):
        segmenter = self.adaptive()
        segmenter.feed(tone(3.8))
        self.assertEqual(segmenter.feed(silence(.78)), [],
                         "Passing four seconds while waiting must not shorten this phrase")
        chunks = segmenter.feed(silence(.02))
        self.assertEqual(len(chunks), 1)
        self.assertAlmostEqual(chunks[0].available_at, 4.6)

    def test_adaptive_long_phrase_emits_at_natural_short_pause(self):
        segmenter = self.adaptive()
        self.assertEqual(segmenter.feed(tone(4)), [])
        self.assertEqual(segmenter.feed(silence(.28)), [])
        chunks = segmenter.feed(silence(.02))
        self.assertEqual(len(chunks), 1)
        self.assertAlmostEqual(chunks[0].available_at, 4.3)
        self.assertAlmostEqual(chunks[0].end, 4.12)
        self.assertEqual(chunks[0].samples.size, 65920)

    def test_adaptive_continuous_speech_keeps_hard_cap_and_all_samples(self):
        segmenter = self.adaptive()
        chunks = segmenter.feed(tone(17.351)) + segmenter.flush()
        self.assertEqual([round(c.duration, 3) for c in chunks], [8, 8, 1.351])
        self.assertTrue(all(c.duration <= 8 for c in chunks))
        for previous, following in zip(chunks, chunks[1:]):
            self.assertAlmostEqual(previous.end, following.start)
        self.assertEqual(sum(c.samples.size for c in chunks), round(17.351 * 16000))

    def test_adaptive_flush_keeps_cursor_but_does_not_leak_long_phrase_state(self):
        segmenter = self.adaptive()
        segmenter.feed(tone(4.351))
        before_pause = segmenter.flush()
        self.assertEqual(len(before_pause), 1)
        self.assertAlmostEqual(before_pause[0].end, 4.351)
        self.assertEqual(segmenter.flush(), [])
        segmenter.feed(tone(1))
        self.assertEqual(segmenter.feed(silence(.78)), [],
                         "A new short phrase after flush must retain the normal pause")
        resumed = segmenter.feed(silence(.02))
        self.assertEqual(len(resumed), 1)
        self.assertAlmostEqual(resumed[0].start, 4.351)
        self.assertAlmostEqual(resumed[0].end, 5.471)
        self.assertAlmostEqual(resumed[0].available_at, 6.151)

    def test_adaptive_parameters_must_be_paired_finite_and_within_normal_limits(self):
        invalid = [
            {"soft_min_seconds": 4}, {"soft_silence_seconds": .3},
            {"soft_min_seconds": 0, "soft_silence_seconds": .3},
            {"soft_min_seconds": -1, "soft_silence_seconds": .3},
            {"soft_min_seconds": 8, "soft_silence_seconds": .3},
            {"soft_min_seconds": 9, "soft_silence_seconds": .3},
            {"soft_min_seconds": 4, "soft_silence_seconds": 0},
            {"soft_min_seconds": 4, "soft_silence_seconds": -.1},
            {"soft_min_seconds": 4, "soft_silence_seconds": .8},
            {"soft_min_seconds": 4, "soft_silence_seconds": 1},
            {"soft_min_seconds": float("nan"), "soft_silence_seconds": .3},
            {"soft_min_seconds": 4, "soft_silence_seconds": float("inf")},
            {"soft_min_seconds": "four", "soft_silence_seconds": .3},
        ]
        for options in invalid:
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.make(max_seconds=8, silence_seconds=.8, **options)


if __name__ == "__main__":
    unittest.main()
