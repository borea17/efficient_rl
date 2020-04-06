from OOMDP_Taxi.agents import Rmax, QLearning
from OOMDP_Taxi.environment import TaxiEnvironment


env = TaxiEnvironment(grid_size=10, mode='classical MDP')
print(env.grid_size)
# agent = QLearning(num_states=env.nS, num_actions=env.nA, gamma=0.95, alpha=1, epsilon=0,
#                   optimistic_init=True, r_max=20)


agent = QLearning(num_states=env.nS, num_actions=env.nA, gamma=0.95, alpha=0.1, epsilon=0.01,
                  optimistic_init=False)  # alpha, epsilon p.33/34 Diuks Dissertation

# agent = Rmax(num_states=env.nS, num_actions=env.nA, gamma=0.95, M=1, max_reward=20)

all_rewards, all_step_times = agent.train(env, num_episodes=int(10e4))
