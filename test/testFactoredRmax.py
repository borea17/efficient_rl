from unittest import TestCase
from efficient_rl.agents import Rmax, FactoredRmax
from efficient_rl.environment import TaxiEnvironment
from efficient_rl.environment.classical_mdp import ClassicalTaxi
from efficient_rl.environment.factored_mdp import FactoredTaxi
import numpy as np
import networkx as nx


class testFactoredRmax(TestCase):

    max_steps = 100
    episodes = 5

    def setUp(self):
        self.factored_envs = [FactoredTaxi(), TaxiEnvironment(grid_size=5, mode='factored MDP')]
        DBNs = self.factored_envs[0].DBNs
        factored_mdp_dict = self.factored_envs[0].factored_mdp_dict
        self.classical_env = ClassicalTaxi()

        self.factored_Rmax = FactoredRmax(M=1, num_states_per_var=[5, 5, 5, 4], num_actions=6,
                                          gamma=0.95, r_max=20, delta=0.01, DBNs=DBNs,
                                          factored_mdp_dict=factored_mdp_dict, env_name='Taxi')
        # comparision between Rmax and FactoredRmax
        self.taxi_Rmax = Rmax(M=1, num_states=500, num_actions=6, gamma=0.95, r_max=20, delta=0.01)
        self.taxi_FactoredRmax_no_reward_DBN = testFactoredRmax.create_no_reward_DBN_FactoredRmax()
        self.taxi_FactoredRmax_no_trans_DBN = testFactoredRmax.create_no_trans_DBN_FactoredRmax()
        return

    def test_agent_main_loop_works_for_some_steps(self):
        print('Test: main loop works for some steps (FactoredRmax)')
        agent = self.factored_Rmax
        for env in self.factored_envs:
            print(' ', env)
            agent.reset()
            env.reset()

            rewards, step_times = agent.main(env, max_steps=testFactoredRmax.max_steps)
        return

    def test_useless_actions_do_not_occur_in_deterministic_factored_Rmax(self):
        print('Test: useless actions do not occur in deterministic factored Rmax')
        agent = self.factored_Rmax
        for env in self.factored_envs:
            print(' ', env)
            agent.reset()
            env.reset()

            done = False
            state = env.s

            last_state, last_action = None, None
            for i_step in range(testFactoredRmax.max_steps):
                action = agent.step(state)

                if np.all(last_state == state):  # last action had no effect => bad action
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
        print('Test: predictions are correct in factored Rmax')
        agent = self.factored_Rmax
        for env in self.factored_envs:
            print(' ', env)
            agent.reset()
            env.reset()

            done = False
            state = env.s
            for i_step in range(testFactoredRmax.max_steps):
                action = agent.step(state)
                new_state, reward, done, _ = env.step(action)
                if not agent.transition_learner_can_predict(state, action):
                    agent.add_experience_to_transition_learner(state, action, new_state)
                else:
                    transition_probs = agent.predict_transition_probs(state, action)
                    pred_new_flat_s = np.argwhere(transition_probs == 1)[0][0]
                    pred_new_state = env.factored_mdp_dict['flat_to_factored_map'][pred_new_flat_s]
                    self.assertTrue(np.all(pred_new_state == new_state))
                if not agent.reward_learner_can_predict(state, action):
                    agent.add_experience_to_reward_learner(state, action, reward)
                else:
                    expected_immediate_reward = agent.predict_expected_immediate_reward(state, action)
                    self.assertTrue(expected_immediate_reward == reward)
                agent.update_emp_MDP(state, action)
                # for M=1, predictions should be possible now:
                if agent.transition_learner_can_predict(state, action) and \
                   agent.reward_learner_can_predict(state, action):
                    # assert transition prediction
                    transition_probs = agent.predict_transition_probs(state, action)
                    pred_new_flat_s = np.argwhere(transition_probs == 1)[0][0]
                    pred_new_state = env.factored_mdp_dict['flat_to_factored_map'][pred_new_flat_s]
                    self.assertTrue(np.all(pred_new_state == new_state))
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
        print('Test: dropoff and pickup do not occur twice in non reward state (FactoredRmax)')
        # this only holds for deterministic FactoredRmax (determinisc environments)
        agent = self.factored_Rmax

        visited_non_reward_dropoff_states = []
        visited_non_reward_pickup_states = []

        for env in self.factored_envs:
            agent.reset()
            env.reset()

            print(' ', env)

            visited_non_reward_dropoff_states.clear()
            visited_non_reward_pickup_states.clear()

            state = env.s
            for i_step in range(testFactoredRmax.max_steps):
                action = agent.step(state)

                if action == 4:  # ensure that corresponding non reward state has not been seen
                    self.assertTrue(state not in visited_non_reward_pickup_states)
                if action == 5:
                    self.assertTrue(state not in visited_non_reward_dropoff_states)

                new_state, reward, done, _ = env.step(action)

                if action == 4 and reward == -10:  # bad Pickup action
                    visited_non_reward_pickup_states.append(state)
                if action == 5 and reward == -10:  # bad Dropfoff action
                    visited_non_reward_dropoff_states.append(state)

                if not agent.transition_learner_can_predict(state, action):
                    agent.add_experience_to_transition_learner(state, action, new_state)
                if not agent.reward_learner_can_predict(state, action):
                    agent.add_experience_to_reward_learner(state, action, reward)

                agent.update_emp_MDP(state, action)

                state = new_state
                if done:
                    break
        return

    def test_Rmax_and_FactoredRmax_do_the_same_thing_when_no_specific_DBN_for_reward_learner(self):
        print('Test Rmax and FactoredRmax do the same when no specific DBN for reward learner')
        RmaxAgent = self.taxi_Rmax
        FactoredRmaxAgent = self.taxi_FactoredRmax_no_reward_DBN
        env, factoredEnv = self.classical_env, self.factored_envs[0]

        RmaxAgent.reset()
        FactoredRmaxAgent.reset()

        for episode in range(testFactoredRmax.episodes):
            # reset environments
            env.reset()
            factoredEnv.reset()
            # set to same start state
            flat_state = env.s
            taxi_row, taxi_col, pass_loc, dest_idx = env.decode(flat_state)
            factored_state = factoredEnv.encode(taxi_row, taxi_col, pass_loc, dest_idx)

            print('Episode ', episode + 1, '/', testFactoredRmax.episodes)

            for step in range(testFactoredRmax.max_steps):
                q_optimal_factored = FactoredRmaxAgent.compute_near_optimal_value_function()
                q_optimal = RmaxAgent.compute_near_optimal_value_function()

                self.assertTrue(np.linalg.norm(q_optimal_factored - q_optimal) < 1e-5)

                action = np.argmax(q_optimal_factored[flat_state])
                flat_new_state, reward, done, _ = env.step(action)
                factored_new_state, factored_reward, factored_done, _ = factoredEnv.step(action)

                flat_new_state_expected = \
                    factoredEnv.factored_mdp_dict['factored_to_flat_map'][tuple(factored_new_state)]
                self.assertTrue(flat_new_state == flat_new_state_expected)
                self.assertTrue(reward == factored_reward)
                self.assertTrue(done == factored_done)

                if not FactoredRmaxAgent.reward_learner_can_predict(factored_state, action):
                    FactoredRmaxAgent.add_experience_to_reward_learner(factored_state, action, reward)
                if not FactoredRmaxAgent.transition_learner_can_predict(factored_state, action):
                    FactoredRmaxAgent.add_experience_to_transition_learner(factored_state, action,
                                                                           factored_new_state)
                FactoredRmaxAgent.update_emp_MDP(factored_state, action)

                if not RmaxAgent.reward_learner_can_predict(flat_state, action):
                    RmaxAgent.add_experience_to_reward_learner(flat_state, action, reward)
                if not RmaxAgent.transition_learner_can_predict(flat_state, action):
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
        print('Test Rmax and FactoredRmax do the same when no specific DBN for transition learner')
        RmaxAgent = self.taxi_Rmax
        FactoredRmaxAgent = self.taxi_FactoredRmax_no_trans_DBN
        env, factoredEnv = self.classical_env, self.factored_envs[0]

        RmaxAgent.reset()
        FactoredRmaxAgent.reset()

        for episode in range(testFactoredRmax.episodes):
            # reset environments
            env.reset()
            factoredEnv.reset()
            # set to same start state
            flat_state = env.s
            taxi_row, taxi_col, pass_loc, dest_idx = env.decode(flat_state)
            factored_state = factoredEnv.encode(taxi_row, taxi_col, pass_loc, dest_idx)

            print('Episode ', episode + 1, '/', testFactoredRmax.episodes)

            for step in range(testFactoredRmax.max_steps):
                q_optimal_factored = FactoredRmaxAgent.compute_near_optimal_value_function()
                q_optimal = RmaxAgent.compute_near_optimal_value_function()

                self.assertTrue(np.linalg.norm(q_optimal_factored - q_optimal) < 1e-5)

                action = np.argmax(q_optimal_factored[flat_state])
                flat_new_state, reward, done, _ = env.step(action)
                factored_new_state, factored_reward, factored_done, _ = factoredEnv.step(action)

                flat_new_state_expected = \
                    factoredEnv.factored_mdp_dict['factored_to_flat_map'][tuple(factored_new_state)]
                self.assertTrue(flat_new_state == flat_new_state_expected)
                self.assertTrue(reward == factored_reward)
                self.assertTrue(done == factored_done)

                if not FactoredRmaxAgent.reward_learner_can_predict(factored_state, action):
                    FactoredRmaxAgent.add_experience_to_reward_learner(factored_state, action, reward)
                if not FactoredRmaxAgent.transition_learner_can_predict(factored_state, action):
                    FactoredRmaxAgent.add_experience_to_transition_learner(factored_state, action,
                                                                           factored_new_state)
                FactoredRmaxAgent.update_emp_MDP(factored_state, action)

                if not RmaxAgent.reward_learner_can_predict(flat_state, action):
                    RmaxAgent.add_experience_to_reward_learner(flat_state, action, reward)
                if not RmaxAgent.transition_learner_can_predict(flat_state, action):
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
    def create_no_reward_DBN_FactoredRmax():
        factored_taxi = FactoredTaxi()
        factored_mdp_dict = factored_taxi.factored_mdp_dict
        num_actions = 6

        no_reward_specific_DBNs = dict()
        no_reward_specific_DBNs['transition'] = factored_taxi.DBNs['transition']
        no_reward_specific_DBNs['reward'] = [factored_taxi.DBNs['reward'][-1]] * num_actions

        taxi_FactoredRmax_no_reward_DBN = FactoredRmax(M=1, num_states_per_var=[5, 5, 5, 4],
                                                       num_actions=6, gamma=0.95, r_max=20,
                                                       delta=0.01, env_name='gym-Taxi',
                                                       DBNs=no_reward_specific_DBNs,
                                                       factored_mdp_dict=factored_mdp_dict)
        return taxi_FactoredRmax_no_reward_DBN

    @staticmethod
    def create_no_trans_DBN_FactoredRmax():
        factored_taxi = FactoredTaxi()
        factored_mdp_dict = factored_taxi.factored_mdp_dict

        no_transition_specific_DBNs = dict()
        no_transition_specific_DBNs['transition'] = \
            testFactoredRmax.create_non_specific_transition_DBNs()
        no_transition_specific_DBNs['reward'] = factored_taxi.DBNs['reward']

        taxi_FactoredRmax_no_transition_DBN = FactoredRmax(M=1, num_states_per_var=[5, 5, 5, 4],
                                                           num_actions=6, gamma=0.95, r_max=20,
                                                           delta=0.01, env_name='gym-Taxi',
                                                           DBNs=no_transition_specific_DBNs,
                                                           factored_mdp_dict=factored_mdp_dict)
        return taxi_FactoredRmax_no_transition_DBN

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
