from rlcard.envs import Env
from rlcard.games.euchre import Game
from rlcard.utils.euchre_utils import ACTION_SPACE, ACTION_LIST
import numpy as np

class EuchreEnv(Env):

    def __init__(self, config):
        self.game = Game()
        self.name = "euchre"

        self.actions = ACTION_LIST
        self.state_shape = [len(self.actions)]
        super().__init__(config)

    def _extract_state(self, state):
        def vec(s):
            suit = {"C":0, "D":1, "H":2, "S":3}
            rank = {"9":9, "T":10, "J":11, "Q":12, "K":13, "A":14}
            if len(s)==1:
                return np.array([ suit[s[0]] ])
            else:
                return np.array([ suit[s[0]], rank[s[1]] ])

        state['legal_actions'] = self._get_legal_actions()
        state['raw_legal_actions'] = self.game.get_legal_actions()

        obs = []
        if state['trump'] is not None:
            obs += [ vec(state['trump']) ]
        else:
            obs += [ np.array([-1]) ]
        if state['flipped'] is not None:
            obs += [ vec(state['flipped']) ]
        else:
            obs += [ np.array([-1,-1]) ]
        if state['lead_suit'] is not None:
            obs += [ vec(state['lead_suit']) ]
        else:
            obs += [ np.array([-1]) ]
        obs += [ vec(e) for e in state['hand'] ]
        obs += [ np.zeros(2*(6-len(state['hand'])))-1 ]
        obs += [ vec(e.get_index()) for e in state['center'] ]
        obs += [ np.zeros(2*(4-len(state['center'])))-1 ]
        obs += [ state['seen'] ]
        state['obs'] = np.hstack(obs)
        
        return state

    def _decode_action(self, action_id):
        return ACTION_LIST[action_id]

    def _get_legal_actions(self):
        legal_actions = self.game.get_legal_actions()
        legal_ids = [ACTION_SPACE[action] for action in legal_actions]
        return legal_ids

    def get_payoffs(self):
        return self.game.get_payoffs()