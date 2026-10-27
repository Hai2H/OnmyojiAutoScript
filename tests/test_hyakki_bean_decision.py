import unittest
import math

from tasks.Hyakkiyakou.agent.focus import Focus
from tasks.Hyakkiyakou.shikigami_table import SHIKIGAMI_TABLE
from tasks.Hyakkiyakou.slave.hya_slave import HyaBuff


class HyakkiBeanDecisionTest(unittest.TestCase):
    def setUp(self):
        self.class_id = next(
            class_id for class_id, row in SHIKIGAMI_TABLE.items()
            if row['rarity'] == 'SSR'
        )
        self.tracks = [(1, self.class_id, .99, 640, 360, 80, 120, 0.)]
        self.focus = Focus(self.tracks[0])
        self.focus.set_omega(.7)
        self.strategy = {'invite_friend': False}

    def test_probability_up_allows_rapid_throw_without_waiting(self):
        state = [101, 250, 36, 10, HyaBuff.BUFF_STATE6]
        action = self.focus.decision(self.tracks, self.strategy, state)
        self.assertTrue(action[2])

    def test_probability_up_keeps_aim_on_rare_target_over_buff(self):
        tracks = [*self.tracks, (2, 5, .99, 100, 200, 40, 50, 0.)]
        state = [101, 250, 36, 10, HyaBuff.BUFF_STATE6]
        action = self.focus.decision(tracks, self.strategy, state)
        self.assertTrue(action[2])
        self.assertTrue(math.isclose(action[0], 600))

    def test_rare_target_without_probability_up_stays_blocked(self):
        state = [1000, 250, 36, 10, HyaBuff.BUFF_STATE0]
        action = self.focus.decision(self.tracks, self.strategy, state)
        self.assertFalse(action[2])

    def test_freeze_blocks_rapid_throw(self):
        state = [101, 250, 36, 10, HyaBuff.BUFF_STATE6]
        action = self.focus.decision(self.tracks, self.strategy, state, freeze=True)
        self.assertFalse(action[2])

    def test_ur_does_not_use_ssr_sp_rapid_mode(self):
        class_id = next(
            class_id for class_id, row in SHIKIGAMI_TABLE.items()
            if row['rarity'] == 'UR'
        )
        tracks = [(1, class_id, .99, 640, 360, 80, 120, 0.)]
        focus = Focus(tracks[0])
        focus.set_omega(.7)
        state = [101, 250, 36, 10, HyaBuff.BUFF_STATE6]
        action = focus.decision(tracks, self.strategy, state)
        self.assertFalse(action[2])


if __name__ == '__main__':
    unittest.main()
