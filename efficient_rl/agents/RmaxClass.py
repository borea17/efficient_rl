from efficient_rl.agents import RmaxBaseAgent
import numpy as np


class Rmax(RmaxBaseAgent):

    def __init__(self, num_states, num_actions, gamma, M, r_max, env_name='Taxi', delta=0.01):
        super().__init__(num_states, num_actions, gamma, M, r_max, env_name, delta)
        # initialize transition and reward learner
        self.state_action_counter_r = np.zeros([num_states, num_actions])
        self.emp_total_reward = np.zeros([num_states, num_actions])
        self.state_action_counter_t = np.zeros([num_states, num_actions])
        self.transition_count = np.zeros([num_actions, num_states, num_states])
        return

    def step(self, state, deterministic=False):
        # Compute a near optimal value function (blowing up here)
        q_optimal = self.compute_near_optimal_value_function()
        # Observe current state and pick action greedily
        if deterministic:
            # pick first action that is optimal
            action = np.argmax(q_optimal[state])
        else:
            # allow for random decision if multiple actions are optimal
            best_actions = np.argwhere(q_optimal[state] == np.amax(q_optimal[state]))
            action = best_actions[np.random.randint(len(best_actions))][0]
        return action

    def update_emp_MDP(self, state, action):
        # empirical distributions are already updated in add_experience
        # if either reward learner or transition learner cannot predict,
        # set distributions back to init
        if not self.reward_learner_can_predict(state, action) or \
           not self.transition_learner_can_predict(state, action):
            transition_probs = np.zeros([len(self.states)])
            transition_probs[state] = 1
            self.emp_transition_dist[action, state] = transition_probs
            self.emp_reward_dist[state, action] = self.r_max
        return

    def add_experience_to_reward_learner(self, state, action, reward):
        # update counts
        self.state_action_counter_r[state, action] += 1
        self.emp_total_reward[state, action] += reward
        # update empirical distribution
        self.emp_reward_dist[state, action] = \
            self.emp_total_reward[state, action] / self.state_action_counter_r[state, action]
        return

    def reward_learner_can_predict(self, state, action):
        return self.state_action_counter_r[state, action] >= self.M

    def predict_expected_immediate_reward(self, state, action):
        expected_immediate_reward = self.emp_reward_dist[state, action]
        return expected_immediate_reward

    def add_experience_to_transition_learner(self, state, action, new_state):
        # update counts
        self.state_action_counter_t[state, action] += 1
        self.transition_count[action, state, new_state] += 1
        # update empirical distribution
        self.emp_transition_dist[action, state, :] = \
            self.transition_count[action, state, :] / self.state_action_counter_t[state, action]
        return

    def transition_learner_can_predict(self, state, action):
        return self.state_action_counter_t[state, action] >= self.M

    def predict_transition_probs(self, state, action):
        transition_probs = self.emp_transition_dist[action, state, :]
        return transition_probs

    def reset(self):
        super().reset()
        num_states, num_actions = len(self.states), len(self.actions)
        self.state_action_counter_r = np.zeros([num_states, num_actions])
        self.emp_total_reward = np.zeros([num_states, num_actions])
        self.state_action_counter_t = np.zeros([num_states, num_actions])
        self.transition_count = np.zeros([num_actions, num_states, num_states])
        return
