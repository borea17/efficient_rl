from unittest import TestCase
from efficient_rl.agents import DOORmax
from efficient_rl.environment import TaxiEnvironment
from efficient_rl.environment.oo_mdp import OOTaxi
import numpy as np


class testDOORmax(TestCase):

    max_steps = 100
    episodes = 5

    def setUp(self):
        effect_types = ['addition', 'multiplication', 'assignment']
        self.oo_envs = [OOTaxi(), TaxiEnvironment(grid_size=5, mode='OO MDP')]
        self.agents = [DOORmax(nS=500, nA=6, num_atts=7, gamma=0.95,
                               r_max=20, env_name='gym-Taxi', k=3, delta=0.01,
                               eff_types=effect_types, oo_mdp_dict=self.oo_envs[0].oo_mdp_dict),
                       DOORmax(nS=500, nA=6, num_atts=7, gamma=0.95, r_max=20,
                               env_name='Taxi', k=3, delta=0.01, eff_types=effect_types,
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

    def test_useless_actions_do_not_occur_in_DOORmax(self):
        print('Test: useless actions do not occur in DOORmax')
        for agent, env in zip(self.agents, self.oo_envs):
            print(' ', env)

            agent.reset()
            env.reset()

            done = False
            state = env.s

            last_state, last_action = [None], [None]
            for i_step in range(testDOORmax.max_steps):
                action = agent.step(state)

                if np.all(last_state[0] == state[0]):  # last action had no effect => bad action
                    # taking the same action would be useless
                    self.assertTrue(last_action != action)
                new_state, reward, done, _ = env.step(action)
                if not agent.transition_learner_can_predict(state, action):
                    agent.add_experience_to_transition_learner(state, action, new_state)
                if not agent.reward_learner_can_predict(state, action):
                    agent.add_experience_to_reward_learner(state, action, reward)
                agent.update_emp_MDP(state, action)

                last_state, last_action = state, action

                state = new_state
                if done:
                    break
        return

    def test_predictions_are_correct(self):
        print('Test: predictions are correct in DOORmax')
        for agent, env in zip(self.agents, self.oo_envs):
            print(' ', env)

            agent.reset()
            env.reset()

            done = False
            state = env.s
            for i_step in range(testDOORmax.max_steps):
                action = agent.step(state)
                new_state, reward, done, _ = env.step(action)
                if not agent.transition_learner_can_predict(state, action):
                    agent.add_experience_to_transition_learner(state, action, new_state)
                else:
                    new_state_pred = agent.predict_transition(state, action)
                    self.assertTrue(np.all(new_state_pred == new_state[0]))
                if not agent.reward_learner_can_predict(state, action):
                    agent.add_experience_to_reward_learner(state, action, reward)
                else:
                    expected_immediate_reward = agent.predict_expected_immediate_reward(state, action)
                    self.assertTrue(expected_immediate_reward == reward)
                agent.update_emp_MDP(state, action)
                # predictions should be possible now:
                if agent.transition_learner_can_predict(state, action) and \
                   agent.reward_learner_can_predict(state, action):
                    # assert transition prediction
                    new_state_pred = agent.predict_transition(state, action)
                    self.assertTrue(np.all(new_state_pred == new_state[0]))
                    # assert reward prediction
                    expected_immediate_reward = agent.predict_expected_immediate_reward(state, action)
                    self.assertTrue(expected_immediate_reward == reward)
                else:
                    self.assertTrue(1 == 0)
                state = new_state

                if done:
                    break
        return

    def test_dropoff_and_pickup_do_not_occur_twice_in_non_reward_state(self):
        print('Test: dropoff and pickup do not occur twice in non reward state (DOORmax)')
        for agent, env in zip(self.agents, self.oo_envs):
            print(' ', env)

            agent.reset()
            env.reset()

            visited_non_reward_dropoff_states = []
            visited_non_reward_pickup_states = []

            state = env.s
            for i_step in range(testDOORmax.max_steps):
                action = agent.step(state)

                if action == 4:  # ensure that corresponding non reward state has not been seen
                    self.assertTrue(state[0] not in visited_non_reward_pickup_states)
                if action == 5:
                    self.assertTrue(state[0] not in visited_non_reward_dropoff_states)

                new_state, reward, done, _ = env.step(action)

                if action == 4 and reward == -10:  # bad Pickup action
                    visited_non_reward_pickup_states.append(state[0])
                if action == 5 and reward == -10:  # bad Dropfoff action
                    visited_non_reward_dropoff_states.append(state[0])

                if not agent.transition_learner_can_predict(state, action):
                    agent.add_experience_to_transition_learner(state, action, new_state)
                if not agent.reward_learner_can_predict(state, action):
                    agent.add_experience_to_reward_learner(state, action, reward)

                agent.update_emp_MDP(state, action)

                state = new_state
                if done:
                    break
        return
