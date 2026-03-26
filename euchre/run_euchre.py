import rlcard
from rlcard.agents.dqn_agent_pytorch import DQNAgent
from rlcard.agents.euchre_rule_agent import EuchreRuleAgent

env = rlcard.make('euchre', config={'num_players': 4})

# Players 0 & 2 are on the same team; players 1 & 3 are on the same team.
# DQN agents (team 0&2) vs Rule-based agents (team 1&3).
dqn0 = DQNAgent(scope='dqn0', action_num=54, state_shape=[48], mlp_layers=[64, 64])
dqn1 = DQNAgent(scope='dqn1', action_num=54, state_shape=[48], mlp_layers=[64, 64])

agents = [dqn0, EuchreRuleAgent(), dqn1, EuchreRuleAgent()]
env.set_agents(agents)

num_games = 10
total_payoffs = {i: 0 for i in range(4)}

for game in range(num_games):
    trajectories, payoffs = env.run(is_training=False)
    for player, score in payoffs.items():
        total_payoffs[player] += score
    print(f'Game {game + 1}: {payoffs}')

print(f'\nAggregate payoffs over {num_games} games:')
print(f'  DQN team  (players 0 & 2): {total_payoffs[0] + total_payoffs[2]}')
print(f'  Rule team (players 1 & 3): {total_payoffs[1] + total_payoffs[3]}')