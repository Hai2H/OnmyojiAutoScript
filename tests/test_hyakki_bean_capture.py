import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tasks.Hyakkiyakou.bean_capture import BeanCapture


class HyakkiBeanCaptureTest(unittest.TestCase):
    def test_capture_serializes_numpy_boolean_state_and_action(self):
        with tempfile.TemporaryDirectory() as temporary:
            capture = BeanCapture(root=temporary, interval_ms=200, prelabel=False)
            image = np.zeros((8, 8, 3), dtype=np.uint8)

            capture._tick(
                image,
                [],
                phase='running',
                state=[np.bool_(True)],
                action=[np.bool_(False)],
            )

            frame_log = Path(capture.folder) / 'frames.jsonl'
            record = json.loads(frame_log.read_text(encoding='utf-8').splitlines()[0])
            self.assertEqual(record['state'], [True])
            self.assertEqual(record['action'], [False])
            self.assertTrue((Path(capture.folder) / 'raw' / 'frame_000001.jpg').is_file())


if __name__ == '__main__':
    unittest.main()
