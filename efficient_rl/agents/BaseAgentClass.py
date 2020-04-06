import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class BaseAgent:

    def __init__(self, num_states, num_actions, gamma, env_name):
        self.states = np.arange(num_states)
        self.actions = np.arange(num_actions)
        self.gamma = gamma
        self.env_name = env_name
        if self.env_name == 'Taxi':  # self written taxi environment
            # Experimental Setup (p.31 Diuks Dissertation)
            # <taxi (x, y) loc, passenger loc, passenger dest>
            self.PROBES = [[(2, 2), 'Y', 'R'],
                           [(2, 2), 'Y', 'G'],
                           [(2, 2), 'Y', 'B'],
                           [(2, 2), 'R', 'B'],
                           [(0, 4), 'Y', 'R'],
                           [(0, 3), 'B', 'G']]
        elif self.env_name == 'gym-Taxi':  # gym taxi extension
            # different x,y orientation:
            self.PROBES = [[(2, 2), 'Y', 'R', 11],
                           [(2, 2), 'Y', 'G', 7],
                           [(2, 2), 'Y', 'B', 8],
                           [(2, 2), 'R', 'B', 8],
                           [(0, 0), 'Y', 'R', 11],  # corresponds to <(0, 4), 'Y', 'R'> from Diuk
                           [(1, 0), 'B', 'G', 8]]   # corresponds to <(0, 3), 'B', 'G'> from Diuk
        else:
            raise NotImplementedError
        return

    def train(self, env, max_episodes=100, max_steps=100):
        all_rewards, all_step_times = [], []
        for i_episode in range(max_episodes):
            env.reset()
            rewards, step_times = self.main(env, max_steps, deterministic=False)

            all_rewards.append(np.sum(rewards))
            all_step_times.extend(step_times)
            if i_episode % 100 == 0:
                print('Episode: {}, Reward: {}, Avg_Step: {}'.format(i_episode, all_rewards[-1],
                                                                     np.mean(step_times)))
            optimum_accomplished = self.evaluate_on_probes(env, max_steps)
            if optimum_accomplished:
                break
        if not optimum_accomplished:
            print("TRAINING DID NOT CONVERGE")
        return all_rewards, all_step_times

    def play(self, env, max_steps=100, deterministic=True):
        all_rewards = []
        state = env.s
        for i_step in range(max_steps):
            action = self.step(state, deterministic)
            new_state, reward, done, _ = env.step(action)

            all_rewards.append(reward)
            state = new_state
            if done:
                break
        return all_rewards

    def evaluate_on_probes(self, env, max_steps=200):
        if self.env_name == 'Taxi':
            if env.grid_size == 5:
                max_scores = [11, 7, 8, 8, 11, 8]
            elif env.grid_size == 10:
                max_scores = [4, -1, 1, -3, 8, 0]
            else:
                raise NotImplementedError

            # scores = []
            for probe, max_score in zip(self.PROBES, max_scores):
                taxi_x, taxi_y = probe[0][0], probe[0][1]
                pass_loc = env.POSITION_NAMES.index(probe[1])
                dest_loc = env.POSITION_NAMES.index(probe[2])

                env.set_state(taxi_y, taxi_x, pass_loc, dest_loc)

                rewards = self.play(env, max_steps, deterministic=True)

                # scores.append(int(sum(rewards)))
                if int(sum(rewards)) != max_score:
                    return False
            # return False
            return True
        elif self.env_name == 'gym-Taxi':
            max_scores = [11, 7, 8, 8, 11, 8]
            gym_encoding = {'R': 0, 'G': 1, 'Y': 2, 'B': 3}
            for probe, max_score in zip(self.PROBES, max_scores):
                taxi_row, taxi_col = probe[0][0], probe[0][1]
                pass_loc = gym_encoding[probe[1]]
                dest_loc = gym_encoding[probe[2]]

                env.s = env.encode(taxi_row, taxi_col, pass_loc, dest_loc)

                rewards = self.play(env, max_steps, deterministic=True)

                if int(sum(rewards)) != max_score:
                    return False
            return True
        else:
            raise NotImplementedError

    def plot_rewards(all_rewards, env, N=100):
        goal_reward = env.spec.reward_threshold
        rolling_mean_reward = pd.Series(all_rewards).rolling(window=N).mean().iloc[N-1:].values
        rolling_std_deviation = pd.Series(all_rewards).rolling(window=N).std().iloc[N-1:].values

        x = np.arange(len(rolling_mean_reward))

        plt.figure()
        plt.plot(rolling_mean_reward, label='rolling mean reward')
        plt.plot(np.array([x[0], x[-1]]), np.array([goal_reward, goal_reward]), label='goal')
        plt.fill_between(x, rolling_mean_reward - rolling_std_deviation,
                         rolling_mean_reward + rolling_std_deviation, alpha=0.2)
        plt.legend()
        plt.xlabel('Episode')
        plt.show()
        return
