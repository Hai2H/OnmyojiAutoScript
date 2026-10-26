import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tasks.Hyakkiyakou.new_detector import DetectionRouter, IouTracker, box_iou, merge_tracks


class DetectorRouterTest(unittest.TestCase):
    def test_iou_tracker_reuses_id_for_overlapping_detection(self) -> None:
        tracker = IouTracker()
        first = tracker.update([{
            'class_id': 265,
            'confidence': .9,
            'bbox': [10, 20, 100, 160],
        }], now=1.0)
        second = tracker.update([{
            'class_id': 265,
            'confidence': .8,
            'bbox': [14, 20, 104, 160],
        }], now=1.2)
        self.assertEqual(first[0][0], second[0][0])
        self.assertGreater(second[0][7], 0)

    def test_assist_keeps_old_and_only_adds_unmatched_new(self) -> None:
        old = [(1, 222, .9, 100, 100, 40, 80, 0)]
        overlapping = (1_000_001, 265, .95, 102, 100, 42, 82, 0)
        separate = (1_000_002, 265, .9, 300, 100, 42, 82, 0)
        merged = merge_tracks(old, [overlapping, separate], 'assist')
        self.assertEqual(merged[0], old[0])
        self.assertEqual(merged[1], separate)
        self.assertEqual(len(merged), 2)

    def test_primary_keeps_new_and_falls_back_to_unmatched_old(self) -> None:
        old = [(1, 222, .9, 100, 100, 40, 80, 0), (2, 100, .8, 400, 100, 40, 80, 0)]
        new = [(1_000_001, 265, .95, 102, 100, 42, 82, 0)]
        merged = merge_tracks(old, new, 'primary')
        self.assertEqual(merged[0], new[0])
        self.assertEqual(merged[1], old[1])

    def test_iou_handles_disjoint_boxes(self) -> None:
        self.assertEqual(box_iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)

    def test_auto_mode_reads_workbench_active_manifest(self) -> None:
        class OldTracker:
            def __call__(self, image, response):
                return [(1, 222, .9, 100, 100, 40, 80, 0)]

        class Remote:
            def detect(self, image):
                return [{
                    'class_id': 265,
                    'confidence': .95,
                    'bbox': [280, 60, 320, 140],
                }]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / 'weights' / 'hyakkiyakou' / 'active.json'
            manifest.parent.mkdir(parents=True)
            manifest.write_text(
                json.dumps({'mode': 'assist', 'model_version': 'test'}),
                encoding='utf-8',
            )
            router = DetectionRouter(
                OldTracker(),
                project_root=root,
                mode='auto',
                remote=Remote(),
            )
            tracks = router(
                np.zeros((720, 1280, 3), dtype=np.uint8),
                [0, 0, False, 10],
            )
            self.assertEqual(router.mode, 'assist')
            self.assertEqual(len(tracks), 2)


if __name__ == '__main__':
    unittest.main()
