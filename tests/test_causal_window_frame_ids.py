from __future__ import annotations

import unittest

from data_provider.temporal_cfd import (
    causal_window_frame_ids,
    causal_window_on_sequence,
    resolve_keyframe_timeline,
)
from layers.Temporal_Physics_Block import CrossStepHistCache


class CausalWindowFrameIdsTests(unittest.TestCase):
    def test_growing_prefix_then_slide_no_pad(self) -> None:
        h = 3
        self.assertEqual(causal_window_frame_ids(0, h), [0])
        self.assertEqual(causal_window_frame_ids(1, h), [0, 1])
        self.assertEqual(causal_window_frame_ids(2, h), [0, 1, 2])
        self.assertEqual(causal_window_frame_ids(3, h), [0, 1, 2, 3])
        self.assertEqual(causal_window_frame_ids(4, h), [1, 2, 3, 4])
        self.assertEqual(causal_window_frame_ids(10, h), [7, 8, 9, 10])
        # Explicit: never left-pad to [0,0,0,0]
        self.assertNotEqual(causal_window_frame_ids(0, h), [0, 0, 0, 0])

    def test_length_bounded(self) -> None:
        h = 3
        for t in range(0, 20):
            ids = causal_window_frame_ids(t, h)
            self.assertLessEqual(len(ids), h + 1)
            self.assertGreaterEqual(len(ids), 1)
            self.assertEqual(ids[-1], t)

    def test_rejects_negative(self) -> None:
        with self.assertRaises(ValueError):
            causal_window_frame_ids(-1, 3)

    def test_sparse_input_timeline_window(self) -> None:
        """uniform_20-style: window along input sequence."""
        timeline = [0, 10, 21, 31, 42, 52]
        h = 3
        self.assertEqual(causal_window_on_sequence(0, h, timeline), [0])
        self.assertEqual(causal_window_on_sequence(1, h, timeline), [0, 10])
        self.assertEqual(causal_window_on_sequence(2, h, timeline), [0, 10, 21])
        self.assertEqual(causal_window_on_sequence(3, h, timeline), [0, 10, 21, 31])
        self.assertEqual(causal_window_on_sequence(4, h, timeline), [10, 21, 31, 42])

    def test_next_input_target(self) -> None:
        """Supervise input t+1: @10 → @21."""
        timeline = resolve_keyframe_timeline(
            n_frames=200,
            valid_frame_ids=[0, 10, 21, 31, 42],
        )
        self.assertEqual(timeline[0], 0)
        self.assertEqual(timeline[1], 10)
        self.assertEqual(timeline[2], 21)
        self.assertEqual(timeline[1 + 1], 21)

    def test_dense_timeline_matches_local_ids(self) -> None:
        timeline = list(range(20))
        h = 3
        for seq_pos in range(20):
            self.assertEqual(
                causal_window_on_sequence(seq_pos, h, timeline),
                causal_window_frame_ids(seq_pos, h),
            )


class CrossStepHistCacheTests(unittest.TestCase):
    def test_reuse_previous_current_on_slide(self) -> None:
        """After [2,3,4,5], step [3,4,5,6] hits 3/4/5 (incl. prev current=5)."""
        import torch

        cache = CrossStepHistCache()
        layer = 0
        for seq_id in (2, 3, 4, 5):
            cache.put_geopt(layer, seq_id, torch.full((1, 4), float(seq_id)))
            cache.put_slice(
                layer,
                seq_id,
                torch.full((1, 2, 4), float(seq_id)),
                torch.full((1, 1, 4, 2), float(seq_id)),
            )
        self.assertEqual(cache.geopt_misses, 0)
        for seq_id in (3, 4, 5):
            hit = cache.get_geopt(layer, seq_id)
            self.assertIsNotNone(hit)
            self.assertEqual(float(hit.mean()), float(seq_id))
        self.assertIsNone(cache.get_geopt(layer, 6))
        self.assertEqual(cache.geopt_hits, 3)
        self.assertEqual(cache.geopt_misses, 1)


if __name__ == "__main__":
    unittest.main()
