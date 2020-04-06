from efficient_rl.agents import BaseAgent
import numpy as np
import time


class RmaxBaseAgent(BaseAgent):

    def __init__(self, total_num_states, num_actions, gamma, M, r_max, env_name, delta):
        super().__init__(total_num_states, num_actions, gamma, env_name)
        self.M = M
        self.r_max = r_max
        self.delta = delta
        # necessary for planning
        self.emp_transition_dist = np.zeros([num_actions, total_num_states, total_num_states])
        self.emp_reward_dist = np.zeros([total_num_states, num_actions])
        self.value_function = np.zeros([total_num_states, 1])
        self.initialize_Rmax_MDP()
        return

    def main(self, env, max_steps=100, deterministic=False):
        # some metrics
        rewards, step_times = [], []
        # start training episode
        state = env.s
        for step in range(max_steps):
            start = time.time()
            # step agent
            action = self.step(state, deterministic)
            # step environment
            new_state, reward, done, _ = env.step(action)
            # add experience to transition learner (only when no predictions can be made)
            if not self.transition_learner_can_predict(state, action):
                self.add_experience_to_transition_learner(state, action, new_state)
            # add experience to reward learner (only when no predictions can be made)
            if not self.reward_learner_can_predict(state, action):
                self.add_experience_to_reward_learner(state, action, reward)
            # update empirical MDP model if possible, otherwise fall back to init
            self.update_emp_MDP(state, action)
            # keep track of metrics
            rewards.append(reward)
            step_times.append(time.time() - start)
            if done:
                break
            state = new_state
        return rewards, step_times

    def compute_near_optimal_value_function(self):
        # use value iteration as Diuk did
        value_function = self.value_function
        P_a, R_a = self.emp_transition_dist, self.emp_reward_dist
        # start iteration loop
        convergence_delta = self.delta + 1
        while convergence_delta > self.delta:
            new_value_function = self.value_iteration_step(P_a, R_a, value_function)
            convergence_delta = np.max(np.abs(new_value_function - value_function))
            value_function = new_value_function
        # store result to speed up computation
        self.value_function = value_function
        # convert optimal value function into optimal action value function
        action_value_function = np.zeros((len(self.states), len(self.actions)))
        for i_action in range(len(self.actions)):
            action_value_function[:, i_action] = R_a[:, i_action] + \
                    self.gamma*(P_a[i_action, :, :] @ value_function).flatten()
        return action_value_function

    def value_iteration_step(self, P_a, R_a, value_function):
        # see David Silver, Lecture 3 p 28
        poss_values = np.zeros([len(self.states), len(self.actions)])
        for i_action in range(len(self.actions)):
            poss_values[:, i_action] = R_a[:, i_action] + \
                self.gamma*(P_a[i_action, :, :] @ value_function).flatten()
        new_value_function = np.expand_dims(np.max(poss_values, axis=1), axis=1)
        return new_value_function

    def initialize_Rmax_MDP(self):
        num_states, num_actions = len(self.states), len(self.actions)

        self.emp_transition_dist[:] = np.eye(num_states)
        self.emp_reward_dist = np.ones([num_states, num_actions]) * self.r_max
        return

    def step(self, state, deterministic=False):
        raise NotImplementedError

    def reward_learner_can_predict(self, state, action):
        raise NotImplementedError

    def predict_expected_immediate_reward(self, state, action):
        raise NotImplementedError

    def add_experience_to_reward_learner(self, state, action, reward):
        raise NotImplementedError

    def transition_learner_can_predict(self, state, action):
        raise NotImplementedError

    def predict_transition_probs(self, state, action):
        raise NotImplementedError

    def add_experience_to_transition_learner(self, state, action, new_state):
        raise NotImplementedError

    def update_emp_MDP(self, state, action):
        raise NotImplementedError
