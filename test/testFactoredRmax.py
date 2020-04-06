from unittest import TestCase
from OOMDP_Taxi.agents import Rmax, FactoredRmax
from OOMDP_Taxi.environment import FactoredTaxi
import gym
import numpy as np
import networkx as nx


class testFactoredRmax(TestCase):

    max_steps = 100
    episodes = 5

    def setUp(self):
        self.env = gym.make("Taxi-v3").env
        num_actions, num_states = self.env.nA, self.env.nS
        self.factoredEnv = FactoredTaxi()
        self.taxi_Rmax = Rmax(M=1, num_states=num_states,
                              num_actions=num_actions,
                              gamma=0.95, max_reward=20, delta=0.01)
        self.taxi_FactoredRmax = FactoredRmax(M=1, num_states_per_state_var=[5, 5, 5, 4],
                                              num_actions=num_actions,
                                              gamma=0.95, max_reward=20, delta=0.01,
                                              DBNs=self.factoredEnv.DBNs)
        no_reward_specific_DBNs = dict()
        no_reward_specific_DBNs['transition'] = self.factoredEnv.DBNs['transition']
        no_reward_specific_DBNs['reward'] = [self.factoredEnv.DBNs['reward'][-1]] * num_actions
        self.taxi_FactoredRmax_no_reward_DBN = FactoredRmax(M=1,
                                                            num_states_per_state_var=[5, 5, 5, 4],
                                                            num_actions=num_actions, gamma=0.95,
                                                            max_reward=20, delta=0.01,
                                                            DBNs=no_reward_specific_DBNs)
        no_transition_specific_DBNs = dict()
        no_transition_specific_DBNs['transition'] = \
            testFactoredRmax.create_non_specific_transition_DBNs()
        no_transition_specific_DBNs['reward'] = self.factoredEnv.DBNs['reward']
        self.taxi_FactoredRmax_no_trans_DBN = FactoredRmax(M=1,
                                                           num_states_per_state_var=[5, 5, 5, 4],
                                                           num_actions=num_actions, gamma=0.95,
                                                           max_reward=20, delta=0.01,
                                                           DBNs=no_transition_specific_DBNs)
        return

    def test_agent_main_loop_works_for_some_steps(self):
        agent = self.taxi_FactoredRmax
        env = self.factoredEnv
        env.reset()
        rewards, step_times = agent.main(env, max_steps=testFactoredRmax.max_steps)
        return

    def test_Rmax_and_FactoredRmax_do_the_same_thing_when_no_specific_DBN_for_reward_learner(self):
        RmaxAgent = self.taxi_Rmax
        FactoredRmaxAgent = self.taxi_FactoredRmax_no_reward_DBN
        env, factoredEnv = self.env, self.factoredEnv

        for episode in range(testFactoredRmax.episodes):
            env.reset()
            factoredEnv.reset()
            flat_state = self.env.s
            factored_state = FactoredRmaxAgent.make_state(flat_state)
            factoredEnv.set_state(factored_state)

            print('Episode ', episode + 1, '/', testFactoredRmax.episodes)
            for step in range(testFactoredRmax.max_steps):
                q_optimal_factored = FactoredRmaxAgent.compute_near_optimal_value_function()
                q_optimal = RmaxAgent.compute_near_optimal_value_function()

                self.assertTrue(np.linalg.norm(q_optimal_factored - q_optimal) < 1e-5)

                action = np.argmax(q_optimal_factored[flat_state])
                flat_new_state, reward, done, _ = env.step(action)
                factored_new_state, factored_reward, factored_done, _ = factoredEnv.step(action)

                self.assertTrue(flat_new_state == FactoredRmaxAgent.make_flat_state(factored_new_state))
                self.assertTrue(reward == factored_reward)
                self.assertTrue(done == factored_done)

                if not FactoredRmaxAgent.reward_learner_can_predict(factored_state, action):
                    FactoredRmaxAgent.add_experience_to_reward_learner(factored_state, action, reward)
                if not FactoredRmaxAgent.transition_learner_can_predict(factored_state, action):
                    FactoredRmaxAgent.add_experience_to_transition_learner(factored_state, action,
                                                                        factored_new_state)
                FactoredRmaxAgent.update_emp_MDP(factored_state, action)

                if RmaxAgent.predict_expected_immediate_reward(flat_state, action) is None:
                    RmaxAgent.add_experience_to_reward_learner(flat_state, action, reward)
                if RmaxAgent.predict_transition_probs(flat_state, action) is None:
                    RmaxAgent.add_experience_to_transition_learner(flat_state, action, flat_new_state)
                RmaxAgent.update_emp_MDP(flat_state, action)

                self.assertTrue(np.linalg.norm(RmaxAgent.emp_reward_dist -
                                               FactoredRmaxAgent.emp_reward_dist) < 1e-5)
                self.assertTrue(np.linalg.norm(RmaxAgent.emp_transition_dist -
                                               FactoredRmaxAgent.emp_transition_dist) < 1e-5)
                if done:
                    break
                flat_state = flat_new_state
                factored_state = factored_new_state
        return

    def test_Rmax_and_FactoredRmax_do_the_same_thing_when_no_specific_DBN_for_trans_learner(self):
        RmaxAgent = self.taxi_Rmax
        FactoredRmaxAgent = self.taxi_FactoredRmax_no_trans_DBN
        env, factoredEnv = self.env, self.factoredEnv

        for episode in range(testFactoredRmax.episodes):
            env.reset()
            factoredEnv.reset()
            flat_state = self.env.s
            factored_state = FactoredRmaxAgent.make_state(flat_state)
            factoredEnv.set_state(factored_state)

            print('Episode ', episode + 1, '/', testFactoredRmax.episodes)
            for step in range(testFactoredRmax.max_steps):
                q_optimal_factored = FactoredRmaxAgent.compute_near_optimal_value_function()
                q_optimal = RmaxAgent.compute_near_optimal_value_function()

                self.assertTrue(np.linalg.norm(q_optimal_factored - q_optimal) < 1e-5)

                action = np.argmax(q_optimal_factored[flat_state])
                flat_new_state, reward, done, _ = env.step(action)
                factored_new_state, factored_reward, factored_done, _ = factoredEnv.step(action)

                self.assertTrue(flat_new_state == FactoredRmaxAgent.make_flat_state(factored_new_state))
                self.assertTrue(reward == factored_reward)
                self.assertTrue(done == factored_done)

                if not FactoredRmaxAgent.reward_learner_can_predict(factored_state, action):
                    FactoredRmaxAgent.add_experience_to_reward_learner(factored_state, action, reward)
                if not FactoredRmaxAgent.transition_learner_can_predict(factored_state, action):
                    FactoredRmaxAgent.add_experience_to_transition_learner(factored_state, action,
                                                                        factored_new_state)
                FactoredRmaxAgent.update_emp_MDP(factored_state, action)

                if RmaxAgent.predict_expected_immediate_reward(flat_state, action) is None:
                    RmaxAgent.add_experience_to_reward_learner(flat_state, action, reward)
                if RmaxAgent.predict_transition_probs(flat_state, action) is None:
                    RmaxAgent.add_experience_to_transition_learner(flat_state, action, flat_new_state)
                RmaxAgent.update_emp_MDP(flat_state, action)

                self.assertTrue(np.linalg.norm(RmaxAgent.emp_reward_dist -
                                               FactoredRmaxAgent.emp_reward_dist) < 1e-5)
                self.assertTrue(np.linalg.norm(RmaxAgent.emp_transition_dist -
                                               FactoredRmaxAgent.emp_transition_dist) < 1e-5)
                if done:
                    break
                flat_state = flat_new_state
                factored_state = factored_new_state
        return

    @staticmethod
    def create_non_specific_transition_DBNs():
        dependencies_south = [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]
        dependencies_north = [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]
        dependencies_east = [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]
        dependencies_west = [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]
        dependencies_pickup = [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]
        dependencies_dropoff = [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]
        dependencies = [dependencies_south, dependencies_north, dependencies_east,
                        dependencies_west, dependencies_pickup, dependencies_dropoff]
        transition_DBNs = []
        state_t = ['y_loc', 'x_loc', 'pass_loc', 'dest_loc']
        state_tp1 = [state for state in range(4)]
        for dependency in dependencies:
            G = nx.DiGraph()
            G.add_nodes_from(state_t, bipartite=0)
            G.add_nodes_from(state_tp1, bipartite=1)
            for i_end in range(len(state_tp1)):
                G.nodes[i_end]['dependency'] = dependency[i_end]
                for i_start in dependency[i_end]:
                    G.add_edges_from([(state_t[i_start], state_tp1[i_end])])
            transition_DBNs.append(G)
        return transition_DBNs


