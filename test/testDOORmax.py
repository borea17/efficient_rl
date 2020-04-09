from unittest import TestCase
from efficient_rl.agents import DOORmax
from efficient_rl.environment import TaxiEnvironment
from efficient_rl.environment.oo_mdp import OOTaxi


class testDOORmax(TestCase):

    max_steps = 100
    episodes = 5

    def setUp(self):
        effect_types = ['addition', 'multiplication', 'assignment']
        self.oo_envs = [OOTaxi(), TaxiEnvironment(grid_size=5, mode='OO MDP')]
        self.agents = [DOORmax(num_states=500, num_actions=6, num_atts=7, gamma=0.95,
                               r_max=20, env_name='gym-Taxi', k=3, delta=0.01,
                               effect_types=effect_types, oo_mdp_dict=self.oo_envs[0].oo_mdp_dict),
                       DOORmax(num_states=500, num_actions=6, num_atts=7, gamma=0.95, r_max=20,
                               env_name='Taxi', k=3, delta=0.01, effect_types=effect_types,
                               oo_mdp_dict=self.oo_envs[1].oo_mdp_dict)]
        return

    def test_agent_main_loop_works_for_some_steps(self):
        print('Test: main loop works for some steps (DOORmax)')
        for agent, env in zip(self.agents, self.oo_envs):
            print(' ', env)
            agent.reset()
            env.reset()

            rewards, step_times = agent.main(env, max_steps=testDOORmax.max_steps)
        return
