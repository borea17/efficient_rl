from unittest import TestCase
from OOMDP_Taxi.agents import Rmax
import gym
import numpy as np


class testRmax(TestCase):

    random_MDP = {'M': 2, 'num_states': 50, 'num_actions': 10, 'gamma': 0.9,
                  'max_reward': 5, 'delta': 0.01}
    SEED = 1
    max_steps = 20

    def setUp(self):
        self.env = gym.make("Taxi-v3")
        self.taxi_agent = Rmax(M=2, num_states=500,
                               num_actions=self.env.action_space.n,
                               gamma=0.95, max_reward=20, delta=0.01)
        self.random_agent = Rmax(M=testRmax.random_MDP['M'],
                                 num_states=testRmax.random_MDP['num_states'],
                                 num_actions=testRmax.random_MDP['num_actions'],
                                 gamma=testRmax.random_MDP['gamma'],
                                 max_reward=testRmax.random_MDP['max_reward'],
                                 delta=testRmax.random_MDP['delta'])
        return

    def test_that_use_less_actions_do_not_occur(self):
        pass

    def test_agent_main_loop_works_for_some_steps(self):
        agent = self.taxi_agent
        env = self.env
        env.reset()
        rewards, step_times = agent.main(env, max_steps=testRmax.max_steps)
        return

    def test_predictions_are_correct(self):
        agent = self.taxi_agent
        env = self.env
        env.reset()
        done = False
        state = env.s
        while not done:
            q_optimal = agent.compute_near_optimal_value_function()
            action = np.argmax(q_optimal[state])
            new_state, reward, done, _ = env.step(action)
            if agent.predict_expected_immediate_reward(state, action) is None:
                agent.add_experience_to_reward_learner(state, action, reward)
            else:
                expected_immediate_reward = agent.predict_expected_immediate_reward(state, action)
                self.assertTrue(expected_immediate_reward == reward)
            if agent.predict_transition_probs(state, action) is None:
                agent.add_experience_to_transition_learner(state, action, new_state)
            else:
                transition_probs = agent.predict_transition_probs(state, action)
                predicted_new_state = np.argwhere(transition_probs == 1)
                self.assertTrue(predicted_new_state == new_state)
            state = new_state
        return

    def test_value_iteration_works_as_expected(self):
        np.random.seed(testRmax.SEED)
        agent = self.random_agent
        num_states, num_actions = len(agent.states), len(agent.actions)
        R_a = np.zeros([num_states, num_actions])
        P_a = np.zeros([num_actions, num_states, num_states])
        for i_state in range(num_states):
            R_a[i_state, :] = np.random.randint(agent.r_max, size=num_actions)
            for i_action in range(num_actions):
                rand_trans_prob_unnormalized = np.random.rand(num_states)
                rand_trans_prob_normalized = rand_trans_prob_unnormalized / \
                    rand_trans_prob_unnormalized.sum()
                P_a[i_action, i_state, :] = rand_trans_prob_normalized
        # compute value function using value iteration of agent
        value_function = np.zeros([num_states, 1])
        convergence_delta = agent.delta + 1
        while convergence_delta > agent.delta:
            new_value_function = agent.value_iteration_step(P_a, R_a, value_function)
            convergence_delta = np.max(np.abs(new_value_function - value_function))
            value_function = new_value_function
        # compute value function using loops
        value_function_verify = np.zeros([num_states, 1])
        convergence_delta = agent.delta + 1
        while convergence_delta > agent.delta:
            poss_action_values = np.zeros([num_states, num_actions])
            for i_action in range(num_actions):
                for i_state in range(num_states):
                    weighted_sum = 0
                    for i_new_state in range(num_states):
                        weighted_sum += P_a[i_action, i_state, i_new_state] * \
                            value_function_verify[i_state]
                    poss_action_values[i_state][i_action] = R_a[i_state][i_action] +\
                        agent.gamma*weighted_sum
            new_value_function = np.max(poss_action_values, axis=1)
            convergence_delta = np.max(np.abs(new_value_function - value_function_verify))
            value_function_verify = new_value_function
        # check that both value functions are equal
        distance = np.linalg.norm(value_function_verify - value_function_verify)
        self.assertTrue(distance < 0.001)
        return
