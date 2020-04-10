import gym
import numpy as np
from efficient_rl.agents import Rmax


env = gym.make('Taxi-v3').env
agent = Rmax(M=1, num_states=env.nS, num_actions=env.nA, gamma=0.95, r_max=20, delta=0.01,
             env_name='gym-Taxi')


all_rewards, all_step_times = agent.train(env, max_episodes=1200, max_steps=100)
print("Avg Step Time: {}, Total Num Steps: {}, Total Time: {}".format(np.mean(all_step_times),
                                                                      len(all_step_times),
                                                                      sum(all_step_times)))

