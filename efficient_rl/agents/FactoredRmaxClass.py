from efficient_rl.agents import RmaxBaseAgent
import numpy as np
import itertools


class FactoredRmax(RmaxBaseAgent):

    def __init__(self, num_states_per_var, num_actions, gamma, M, r_max, delta, DBNs, env_name):
        total_num_states = np.prod(num_states_per_var)
        super().__init__(total_num_states, num_actions, gamma, M, r_max, env_name, delta)
        self.num_states_per_state_variable = np.array(num_states_per_var)
        # extract DBNs
        transition_DBNs, reward_DBNs = DBNs['transition'], DBNs['reward']
        # initialize factored reward and transition learner
        self.state_action_count_r, self.factored_total_reward, self.factored_emp_reward = \
            self.create_factored_reward_learner(reward_DBNs)
        self.state_action_count_t, self.factored_transition_count, self.factored_emp_transition = \
            self.create_factored_transition_learner(transition_DBNs)
        # store DBNs for later use
        self.DBNs = DBNs
        # to enhance update procedure
        self.new_states = [self.make_state(flat_state) for flat_state in self.states]
        self.all_possible_state_action_seq = set((state, action) for state, action in
                                                 itertools.product(self.states, self.actions))
        self.updated_state_action_seq = set()
        return

    def step(self, state, deterministic=False):
        flat_state = self.make_flat_state(state)
        # Compute a near optimal value function (blowing up here)
        q_optimal = self.compute_near_optimal_value_function()
        # Observe current state and pick action greedily
        if deterministic:
            # pick first action that is optimal
            action = np.argmax(q_optimal[flat_state])
        else:
            # allow for random decision if multiple actions are optimal
            best_actions = np.argwhere(q_optimal[flat_state] == np.amax(q_optimal[flat_state]))
            action = best_actions[np.random.randint(len(best_actions))][0]
        return action

    def update_emp_MDP(self, state_not_used, action_not_used):
        state_actions_to_check = self.all_possible_state_action_seq - self.updated_state_action_seq
        for flat_state, action in state_actions_to_check:
            state = self.make_state(flat_state)
            # update model only if both transition and reward learner can predict
            if self.reward_learner_can_predict(state, action) and \
               self.transition_learner_can_predict(state, action):
                self.emp_reward_dist[flat_state, action] = \
                    self.predict_expected_immediate_reward(state, action)
                self.emp_transition_dist[action, flat_state] = \
                    self.predict_transition_probs(state, action)
                self.updated_state_action_seq.add((flat_state, action))
        return

    def add_experience_to_transition_learner(self, state, action, new_state):
        # convert state into state that includes parent states (use DBN)
        state_incl_par = self.get_state_including_parents_for_transition(state, action)
        for i_state in range(len(state)):
            # retrieve state-variable that includes dependencies of parents
            state_var_incl_par = state_incl_par[i_state]
            # update count
            self.state_action_count_t[i_state][action][state_var_incl_par] += 1
            # obtain next state-variable value
            new_state_var = new_state[i_state]
            # update transition count
            self.factored_transition_count[i_state][action][state_var_incl_par][new_state_var] += 1
            # update empirical transition count for each state variable
            self.factored_emp_transition[i_state][action][state_var_incl_par][:] = \
                self.factored_transition_count[i_state][action][state_var_incl_par][:] / \
                self.state_action_count_t[i_state][action][state_var_incl_par]
            # ensure that a true probability distribution is obtained
            # assert round(sum(self.factored_emp_transition[i_state][action][state_var_incl_par]), 5) == 1
        return

    def transition_learner_can_predict(self, state, action):
        state_incl_par = self.get_state_including_parents_for_transition(state, action)
        counts = [self.state_action_count_t[i_state][action][state_var_incl_par] for
                  i_state, state_var_incl_par in enumerate(state_incl_par)]
        return all(np.array(counts) >= self.M)

    def predict_transition_probs(self, state, action):
        state_incl_par = self.get_state_including_parents_for_transition(state, action)
        # retrieve transition probabilites for each state variable
        transition_state = []
        for i_state, state_v_inc_par in enumerate(state_incl_par):
            transition_state.append(self.factored_emp_transition[i_state][action][state_v_inc_par])
        # use these probabilites to define flat state transition probabilites (BLOW UP)
        flat_transition_probs = np.zeros(len(self.states))
        for flat_new_state, new_state in enumerate(self.new_states):
            flat_transition_probs[flat_new_state] = np.prod([transition_state[i_state][state_var]
                                                             for i_state, state_var in
                                                             enumerate(new_state)])
        # ensure that true probability distribution is obtained
        # assert round(flat_transition_probs.sum(), 5) == 1
        return flat_transition_probs

    def add_experience_to_reward_learner(self, state, action, reward):
        state_incl_par = self.get_state_including_parents_for_reward(state, action)
        self.state_action_count_r[action][state_incl_par] += 1
        self.factored_total_reward[action][state_incl_par] += reward
        self.factored_emp_reward[action][state_incl_par] = \
            self.factored_total_reward[action][state_incl_par] / \
            self.state_action_count_r[action][state_incl_par]
        return

    def reward_learner_can_predict(self, state, action):
        state_incl_par = self.get_state_including_parents_for_reward(state, action)
        return self.state_action_count_r[action][state_incl_par] >= self.M

    def predict_expected_immediate_reward(self, state, action):
        state_incl_par = self.get_state_including_parents_for_reward(state, action)
        expected_immediate_reward = self.factored_emp_reward[action][state_incl_par]
        return expected_immediate_reward

    def get_state_including_parents_for_transition(self, state, action):
        state_incl_par = []
        transition_DBNs = self.DBNs['transition']
        for i_state, state_var in enumerate(state):
            parents_of_state_variable = transition_DBNs[action].nodes[i_state]['dependency']
            state_var_incl_par = state[parents_of_state_variable[0]]
            for parent in parents_of_state_variable[1:]:
                state_var_incl_par *= self.num_states_per_state_variable[parent]
                state_var_incl_par += state[parent]
            state_incl_par.append(state_var_incl_par)
        return state_incl_par

    def get_state_including_parents_for_reward(self, state, action):
        reward_DBNs = self.DBNs['reward']
        parents_of_state = reward_DBNs[action].nodes['reward']['dependency']
        state_incl_par = state[parents_of_state[0]]
        for parent in parents_of_state[1:]:
            state_incl_par *= self.num_states_per_state_variable[parent]
            state_incl_par += state[parent]
        return state_incl_par

    def make_flat_state(self, state):
        flat_state = state[0]
        for index, state_var in enumerate(state[1:]):
            flat_state *= self.num_states_per_state_variable[index + 1]
            flat_state += state_var
        return flat_state

    def make_state(self, flat_state):
        state = []
        for num_states_in_state_var in self.num_states_per_state_variable[::-1][:-1]:
            state.append(flat_state % num_states_in_state_var)
            flat_state = flat_state // num_states_in_state_var
        state.append(flat_state)
        # assert 0 <= flat_state <= self.num_states_per_state_variable[-1]
        return state[::-1]

    def create_factored_transition_learner(self, transition_DBNs):
        state_action_count_t = np.empty([len(self.num_states_per_state_variable),
                                           len(self.actions)], dtype=object)
        factored_transition_count = np.empty([len(self.num_states_per_state_variable),
                                              len(self.actions)], dtype=object)
        factored_emp_transition = np.empty([len(self.num_states_per_state_variable),
                                            len(self.actions)], dtype=object)

        for action in self.actions:
            DBN = transition_DBNs[action]
            for i_state, num_states in enumerate(self.num_states_per_state_variable):
                parent_states = DBN.nodes[i_state]['dependency']
                num_states_per_par_state = self.num_states_per_state_variable[parent_states]
                num_dep_states = num_states_per_par_state.prod()

                state_action_count_t[i_state][action] = np.zeros([num_dep_states])
                factored_transition_count[i_state][action] = np.zeros([num_dep_states, num_states])
                factored_emp_transition[i_state][action] = np.zeros([num_dep_states, num_states])
        return state_action_count_t, factored_transition_count, factored_emp_transition

    def create_factored_reward_learner(self, reward_DBNs):
        state_action_counter_r = np.empty(len(self.actions), dtype=object)
        factored_total_reward = np.empty(len(self.actions), dtype=object)
        factored_emp_reward = np.empty(len(self.actions), dtype=object)

        for action in self.actions:
            DBN = reward_DBNs[action]
            dependent_states = DBN.nodes['reward']['dependency']

            num_dependent_states = self.num_states_per_state_variable[dependent_states].prod()

            state_action_counter_r[action] = np.zeros([num_dependent_states])
            factored_total_reward[action] = np.zeros([num_dependent_states])
            factored_emp_reward[action] = np.zeros([num_dependent_states])
        return state_action_counter_r, factored_total_reward, factored_emp_reward


if __name__ == '__main__':
    from OOMDP_Taxi.environment import FactoredTaxi
    env = FactoredTaxi()
    DBNs = env.DBNs
    agent = FactoredRmax(num_states_per_state_var=[5, 5, 5, 4],
                         num_actions=6,
                         gamma=0.95,
                         M=1,
                         delta=0.0001,
                         DBNs=DBNs,
                         max_reward=20)
    agent.train(env, 50, 100)
