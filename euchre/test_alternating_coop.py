import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import unittest

from train_alternating_coop import (
    AlternatingCooperativeTrainer,
    AlternatingTrainingConfig,
)


def make_test_config():
    return AlternatingTrainingConfig(
        phase_episodes=2,
        max_rounds=1,
        eval_games=4,
        convergence_tol=0.0,
        patience_rounds=1,
        batch_size=2,
        replay_memory_size=200,
        replay_memory_init_size=2,
        update_target_every=10,
        epsilon_decay_steps=100,
        mlp_layers=(32,),
        seed=7,
    )


class TestAlternatingCoopTrainer(unittest.TestCase):

    def setUp(self):
        self.trainer = AlternatingCooperativeTrainer(make_test_config())

    def test_run_phase_trains_only_active_player(self):
        self.assertEqual(len(self.trainer.agents[0].memory.memory), 0)
        self.assertEqual(len(self.trainer.agents[2].memory.memory), 0)

        phase0 = self.trainer.run_phase(train_player=0, num_episodes=2)
        p0_after_first = len(self.trainer.agents[0].memory.memory)
        p2_after_first = len(self.trainer.agents[2].memory.memory)

        self.assertGreater(p0_after_first, 0)
        self.assertEqual(p2_after_first, 0)
        self.assertIn('eval_win_rate', phase0)
        self.assertIn('eval_win_pct', phase0)
        self.assertIn('eval_wins', phase0)
        self.assertEqual(phase0['eval_games'], 4)

        self.trainer.run_phase(train_player=2, num_episodes=2)
        self.assertEqual(len(self.trainer.agents[0].memory.memory), p0_after_first)
        self.assertGreater(len(self.trainer.agents[2].memory.memory), p2_after_first)

    def test_train_alternates_between_players(self):
        history = self.trainer.train()
        self.assertEqual(len(history), 2)
        self.assertEqual([phase['train_player'] for phase in history], [0, 2])


if __name__ == '__main__':
    unittest.main(verbosity=2)
