import unittest

import numpy as np

from tasks.Hyakkiyakou.agent.agent import Agent
from tasks.Hyakkiyakou.agent.focus import display_label, display_name, is_high_rarity


class NewClassDecisionTest(unittest.TestCase):
    def test_new_sp_ssr_ur_classes_are_high_rarity(self) -> None:
        for class_id in (264, 265, 266, 267):
            with self.subTest(class_id=class_id):
                self.assertTrue(is_high_rarity(class_id))
                self.assertNotEqual(display_label(class_id), display_name(class_id))

    def test_gamma_accepts_new_class_ids_outside_old_oashya_ranges(self) -> None:
        weights = [1., 1., .7, .3, 0., 0., 1.5]
        for class_id in (264, 265, 266, 267):
            with self.subTest(class_id=class_id):
                tracks = [(1, class_id, .95, 640, 360, 80, 120, 0.)]
                z = Agent.gamma(tracks, weights=weights)
                self.assertGreater(float(np.max(z)), 0.0)


if __name__ == '__main__':
    unittest.main()
