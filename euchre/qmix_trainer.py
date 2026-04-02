"""
QMIX training loop for cooperative Euchre (players 0 & 2 as one team).

Sequential-turn handling follows Option 1 from the Phase 2 spec:
  - global state s_t is snapshotted once at the start of each trick
  - both agents' transitions within that trick share the same s_t / s_t'
"""

import numpy as np
from rlcard.envs.euchre import EuchreEnv
from rlcard.agents.dqn_agent_pytorch import DQNAgent, JointMemory

ENV_CONFIG = {
    'allow_step_back': False,
    'allow_raw_data': False,
    'record_action': False,
    'single_agent_mode': False,
    'active_player': 0,
    'seed': None,
}

TEAM_PLAYERS = {0, 2}   # agents controlled by QMIX
OPP_PLAYERS  = {1, 3}   # opponents (rule/random agents)


class QMIXTrainer:
    def __init__(self,
                 agent0,           # DQNAgent for player 0
                 agent2,           # DQNAgent for player 2
                 opp_agents,       # dict {1: agent, 3: agent} for opponents
                 joint_memory,     # JointMemory instance
                 env=None):
        self.agent0 = agent0
        self.agent2 = agent2
        self.opp_agents = opp_agents
        self.joint_memory = joint_memory
        self.env = env or EuchreEnv(ENV_CONFIG)

    def run_episode(self):
        """Play one full hand and populate joint_memory with trick-level transitions.

        Returns the shared reward received by team {0, 2}.
        """
        state, player_id = self.env.game.init_game()
        state = self.env._extract_state(state)

        # Per-trick staging buffers
        trick_global_state = None   # snapshotted once per trick
        pending = {}                # {player_id: (obs, action)} within current trick

        cards_at_trick_start = None

        while not self.env.game.is_over():
            # --- snapshot global state at the start of each trick ---
            cur_hand_size = len(self.env.game.players[0].hand)
            if cards_at_trick_start != cur_hand_size and len(self.env.game.center) == 0:
                trick_global_state = self.env.get_global_state()
                cards_at_trick_start = cur_hand_size
                pending = {}

            obs = state['obs']

            # --- select action ---
            if player_id == 0:
                action = self.agent0.step(state)
            elif player_id == 2:
                action = self.agent2.step(state)
            else:
                action = self.opp_agents[player_id].step(state)

            # store pending transition for team players
            if player_id in TEAM_PLAYERS:
                pending[player_id] = (obs, action)

            # --- step environment ---
            next_state, next_player_id = self.env.step(action)

            # --- when both team players have acted in this trick, save joint transition ---
            if TEAM_PLAYERS.issubset(pending.keys()):
                next_global_state = self.env.get_global_state()
                payoffs = self.env.game.get_payoffs() if self.env.game.is_over() else {}
                reward = payoffs.get(0, 0.0)   # team reward (same for 0 and 2)

                obs1, a1 = pending[0]
                obs2, a2 = pending[2]
                next_obs1 = next_state['obs'] if next_player_id == 0 else \
                            self.env._extract_state(self.env.game.get_state(0))['obs']
                next_obs2 = next_state['obs'] if next_player_id == 2 else \
                            self.env._extract_state(self.env.game.get_state(2))['obs']

                self.joint_memory.save(
                    obs1, obs2, a1, a2,
                    next_obs1, next_obs2,
                    reward,
                    trick_global_state,
                    next_global_state,
                )
                pending = {}

            state = next_state
            player_id = next_player_id

        payoffs = self.env.game.get_payoffs()
        return payoffs.get(0, 0.0)
