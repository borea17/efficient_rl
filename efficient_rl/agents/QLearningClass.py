import numpy as np
import time
from efficient_rl.agents import BaseAgent


class QLearning(BaseAgent):

    def __init__(self, num_states, num_actions, gamma, alpha, epsilon, optimistic_init, env_name,
                 r_max=0):
        super().__init__(num_states, num_actions, gamma, env_name)
        self.optimistic_init = optimistic_init
        if optimistic_init:
            self.r_max = r_max
            self.Q_table = (r_max/(1-self.gamma)) * np.ones([num_states, num_actions])
        else:
            self.Q_table = np.zeros([num_states, num_actions])
        self.alpha = alpha
        self.epsilon = epsilon
        return

    def main(self, env, max_steps=100, deterministic=False):
        # some metrics
        rewards, step_times = [], []
        # start training procedure
        state = env.s
        for step in range(max_steps):
            start = time.time()
            action = self.step(state, deterministic)
            new_state, reward, done, _ = env.step(action)
            self.update(state, action, reward, new_state)
            rewards.append(reward)
            step_times.append(time.time() - start)
            if done:
                break
            state = new_state
        return rewards, step_times

    def step(self, state, deterministic=False):
        # epsilon greedy exploration
        if np.random.rand() < self.epsilon and not deterministic:
            action = np.random.randint(len(self.actions))
        elif not deterministic:
            # allow for random decision if multiple actions are optimal
            best_actions = np.argwhere(self.Q_table[state, :] == np.amax(self.Q_table[state, :]))
            action = best_actions[np.random.randint(len(best_actions))][0]
        else:  # deterministic mode
            action = np.argmax(self.Q_table[state, :])
        return action

    def update(self, state, action, reward, new_state):
        self.Q_table[state, action] = (1 - self.alpha)*self.Q_table[state, action] + \
            self.alpha*(reward + self.gamma * np.max(self.Q_table[new_state, :]))
        return

    def reset(self):
        num_states, num_actions = len(self.states), len(self.actions)
        if self.optimistic_init:
            r_max = self.r_max
            self.Q_table = (r_max/(1-self.gamma)) * np.ones([num_states, num_actions])
        else:
            self.Q_table = np.zeros([num_states, num_actions])
        return
