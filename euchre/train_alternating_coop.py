"""
Alternating cooperative training for Euchre.

Players 0 and 2 are teammates. We alternate best-response phases:

1. Train player 0 while player 2 is frozen to its current greedy policy.
2. Train player 2 while player 0 is frozen to its current greedy policy.
3. Repeat until the joint policy converges or a max number of rounds is hit.

Opponents (players 1 and 3) are fixed rule-based agents.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

import rlcard
from rlcard.agents.dqn_agent_pytorch import DQNAgent
from rlcard.agents.euchre_rule_agent import EuchreRuleAgent


OBS_DIM = 48
ACTION_NUM = 54


@dataclass
class AlternatingTrainingConfig:
    phase_episodes: int = 300
    max_rounds: int = 12
    eval_games: int = 150
    convergence_tol: float = 0.02
    patience_rounds: int = 3
    trick_reward_scale: float = 0.25
    gamma: float = 0.99
    learning_rate: float = 5e-4
    batch_size: int = 32
    replay_memory_size: int = 20000
    replay_memory_init_size: int = 200
    update_target_every: int = 500
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 6000
    mlp_layers: tuple[int, ...] = (128, 128)
    seed: int | None = None


class AlternatingCooperativeTrainer:
    """Alternate training between teammates 0 and 2."""

    def __init__(self, config: AlternatingTrainingConfig):
        self.config = config
        self.env = rlcard.make('euchre', config={'num_players': 4, 'seed': config.seed})
        self.agents = {
            0: self._make_learning_agent('coop_p0'),
            2: self._make_learning_agent('coop_p2'),
        }
        self.opp_agents = {1: EuchreRuleAgent(), 3: EuchreRuleAgent()}
        self.env.set_agents([
            self.agents[0],
            self.opp_agents[1],
            self.agents[2],
            self.opp_agents[3],
        ])
        self.phase_history: List[Dict[str, float]] = []

    @staticmethod
    def _format_pct(value: float) -> str:
        return f"{value * 100:6.2f}%"

    def _print_banner(self) -> None:
        print("=" * 72)
        print("Alternating Cooperative Euchre Training")
        print("  Team learner : players 0 and 2")
        print("  Opponents    : rule-based players 1 and 3")
        print(f"  Phase hands  : {self.config.phase_episodes}")
        print(f"  Max rounds   : {self.config.max_rounds}")
        print(f"  Eval hands   : {self.config.eval_games}")
        print("=" * 72)

    def _print_phase_summary(self, round_idx: int, phase: Dict[str, float]) -> None:
        print(
            f"  Round {round_idx:>2} | Train P{phase['train_player']} | "
            f"train payoff {phase['phase_avg_training_payoff']:+.3f} | "
            f"WIN RATE {self._format_pct(phase['eval_win_rate'])} | "
            f"eval payoff {phase['eval_avg_payoff']:+.3f}"
        )

    def _print_round_summary(self, round_idx: int, p0_phase: Dict[str, float], p2_phase: Dict[str, float]) -> None:
        best_phase = max((p0_phase, p2_phase), key=lambda phase: phase['eval_win_rate'])
        print("-" * 72)
        print(
            f"End of round {round_idx}: current joint WIN RATE "
            f"{self._format_pct(p2_phase['eval_win_rate'])} | "
            f"avg payoff {p2_phase['eval_avg_payoff']:+.3f}"
        )
        print(
            f"  Best phase this round: after training P{best_phase['train_player']} "
            f"the team reached {self._format_pct(best_phase['eval_win_rate'])}"
        )
        print("-" * 72)

    def _make_learning_agent(self, scope: str) -> DQNAgent:
        return DQNAgent(
            scope=scope,
            action_num=ACTION_NUM,
            state_shape=[OBS_DIM],
            mlp_layers=list(self.config.mlp_layers),
            replay_memory_size=self.config.replay_memory_size,
            replay_memory_init_size=self.config.replay_memory_init_size,
            update_target_estimator_every=self.config.update_target_every,
            discount_factor=self.config.gamma,
            epsilon_start=self.config.epsilon_start,
            epsilon_end=self.config.epsilon_end,
            epsilon_decay_steps=self.config.epsilon_decay_steps,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
        )

    @staticmethod
    def _team_tricks(game) -> int:
        return game.score[0] + game.score[2]

    @staticmethod
    def _opp_tricks(game) -> int:
        return game.score[1] + game.score[3]

    def _policy_action(self, player_id: int, state, train_player: int | None = None) -> int:
        if player_id in self.opp_agents:
            action, _ = self.opp_agents[player_id].eval_step(state)
            return action

        if player_id == train_player:
            return self.agents[player_id].step(state)

        action, _ = self.agents[player_id].eval_step(state)
        return action

    def play_training_episode(self, train_player: int) -> float:
        """
        Play one hand while training only `train_player`.

        Rewards are shared team rewards:
        - +/- trick_reward_scale on each trick depending on which team wins it
        - final hand payoff in {-2, -1, 1, 2} at terminal
        """
        active_agent = self.agents[train_player]
        state, player_id = self.env.reset()
        pending_state = None
        pending_action = None
        pending_reward = 0.0
        prev_team_diff = self._team_tricks(self.env.game) - self._opp_tricks(self.env.game)

        while not self.env.is_over():
            if player_id == train_player and pending_state is not None:
                active_agent.feed((
                    pending_state,
                    pending_action,
                    pending_reward,
                    state,
                    False,
                ))
                pending_state = None
                pending_action = None
                pending_reward = 0.0

            action = self._policy_action(player_id, state, train_player=train_player)

            if player_id == train_player:
                pending_state = state
                pending_action = action

            prev_center_len = len(self.env.game.center)
            next_state, next_player_id = self.env.step(action)
            hand_ended = self.env.is_over()
            trick_ended = (prev_center_len == 3 and len(self.env.game.center) == 0)

            if pending_state is not None and trick_ended:
                team_diff = self._team_tricks(self.env.game) - self._opp_tricks(self.env.game)
                pending_reward += self.config.trick_reward_scale * (team_diff - prev_team_diff)
                prev_team_diff = team_diff

            if hand_ended:
                if pending_state is not None:
                    terminal_state = self.env.get_state(train_player)
                    terminal_reward = self.env.game.get_payoffs().get(train_player, 0.0)
                    active_agent.feed((
                        pending_state,
                        pending_action,
                        pending_reward + terminal_reward,
                        terminal_state,
                        True,
                    ))
                break

            state = next_state
            player_id = next_player_id

        return self.env.game.get_payoffs().get(0, 0.0)

    def run_phase(self, train_player: int, num_episodes: int) -> Dict[str, float]:
        payoffs = []
        for _ in range(num_episodes):
            payoffs.append(self.play_training_episode(train_player))

        win_rate, avg_payoff = self.evaluate_joint_policy(self.config.eval_games)
        win_count = int(round(win_rate * self.config.eval_games))
        result = {
            'train_player': train_player,
            'episodes': num_episodes,
            'phase_avg_training_payoff': float(np.mean(payoffs)) if payoffs else 0.0,
            'eval_win_rate': win_rate,
            'eval_win_pct': win_rate * 100.0,
            'eval_wins': win_count,
            'eval_games': self.config.eval_games,
            'eval_avg_payoff': avg_payoff,
        }
        self.phase_history.append(result)
        return result

    def evaluate_joint_policy(self, num_games: int) -> tuple[float, float]:
        wins = 0
        total = 0.0

        for _ in range(num_games):
            state, player_id = self.env.reset()
            while not self.env.is_over():
                action = self._policy_action(player_id, state, train_player=None)
                state, player_id = self.env.step(action)

            payoff = self.env.game.get_payoffs().get(0, 0.0)
            total += payoff
            if payoff > 0:
                wins += 1

        return wins / num_games, total / num_games

    def train(self) -> List[Dict[str, float]]:
        stagnant_rounds = 0
        last_round_eval = None
        self._print_banner()

        for round_idx in range(1, self.config.max_rounds + 1):
            print(f"\nRound {round_idx}/{self.config.max_rounds}")

            round_phases = []
            for train_player in (0, 2):
                phase = self.run_phase(train_player, self.config.phase_episodes)
                round_phases.append(phase)
                self._print_phase_summary(round_idx, phase)

            self._print_round_summary(round_idx, round_phases[0], round_phases[1])

            round_eval = self.phase_history[-1]['eval_avg_payoff']
            if last_round_eval is not None:
                if abs(round_eval - last_round_eval) <= self.config.convergence_tol:
                    stagnant_rounds += 1
                else:
                    stagnant_rounds = 0

                if stagnant_rounds >= self.config.patience_rounds:
                    print(
                        "Stopping early: joint payoff change stayed within "
                        f"{self.config.convergence_tol:.3f} for {self.config.patience_rounds} rounds."
                    )
                    break

            last_round_eval = round_eval

        return self.phase_history

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'config': asdict(self.config),
            'agent0': self.agents[0].get_state_dict(),
            'agent2': self.agents[2].get_state_dict(),
            'history': self.phase_history,
        }, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Alternating cooperative Euchre training')
    parser.add_argument('--phase-episodes', type=int, default=300)
    parser.add_argument('--max-rounds', type=int, default=12)
    parser.add_argument('--eval-games', type=int, default=150)
    parser.add_argument('--convergence-tol', type=float, default=0.02)
    parser.add_argument('--patience-rounds', type=int, default=3)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--checkpoint', type=str, default='checkpoints/alternating_coop.pt')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AlternatingTrainingConfig(
        phase_episodes=args.phase_episodes,
        max_rounds=args.max_rounds,
        eval_games=args.eval_games,
        convergence_tol=args.convergence_tol,
        patience_rounds=args.patience_rounds,
        seed=args.seed,
    )

    trainer = AlternatingCooperativeTrainer(config)
    history = trainer.train()
    trainer.save_checkpoint(args.checkpoint)

    final_win_rate, final_avg_payoff = trainer.evaluate_joint_policy(config.eval_games)
    print("\n" + "=" * 72)
    print("Final greedy evaluation")
    print(f"  WIN RATE   : {final_win_rate * 100:6.2f}% ({int(round(final_win_rate * config.eval_games))}/{config.eval_games})")
    print(f"  Avg payoff : {final_avg_payoff:+.3f}")
    print(f"  Phases run : {len(history)}")
    print(f"  Checkpoint : {args.checkpoint}")
    print("=" * 72)


if __name__ == '__main__':
    main()
