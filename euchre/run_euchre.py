import rlcard
from rlcard.agents.euchre_rule_agent import EuchreRuleAgent

env = rlcard.make('euchre', config={'num_players': 4})

agents = [EuchreRuleAgent() for _ in range(4)]
env.set_agents(agents)

trajectories, payoffs = env.run(is_training=False)
print('Payoffs:', payoffs)