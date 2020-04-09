from prettytable import PrettyTable
from efficient_rl.agents import Rmax, FactoredRmax, QLearning
from efficient_rl.environment import TaxiEnvironment
import numpy as np

grid_size = 10
max_episodes = 5000
max_steps = 100

agent_names = ['Rmax', 'Factored Rmax', 'Q Learning', 'Q Learning (optimistic initialization)']
envs = [TaxiEnvironment(grid_size=grid_size, mode='classical MDP'),
        TaxiEnvironment(grid_size=grid_size, mode='factored MDP'),
        TaxiEnvironment(grid_size=grid_size, mode='classical MDP'),
        TaxiEnvironment(grid_size=grid_size, mode='classical MDP')]
agents = [Rmax(M=1, num_states=envs[0].nS, num_actions=envs[0].nA, gamma=0.95, r_max=20,
               delta=0.01, env_name='Taxi'),
          FactoredRmax(M=1, num_states_per_var=envs[1].num_states_per_var, num_actions=envs[1].nA,
                       gamma=0.95, r_max=envs[1].r_max, delta=0.01, DBNs=envs[1].DBNs,
                       env_name='Taxi'),
          QLearning(num_states=envs[2].nS, num_actions=envs[2].nA, gamma=0.95, alpha=0.1,
                    epsilon=0.6, optimistic_init=False, env_name='Taxi'),  # p.33/34 Diuks Diss
          QLearning(num_states=envs[3].nS, num_actions=envs[3].nA, gamma=0.95, alpha=1, epsilon=0,
                    optimistic_init=True, r_max=envs[3].r_max, env_name='Taxi')]


statistics = {}

index = 0
# for agent, env, agent_name in zip(agents, envs, agent_names):
for agent, env, agent_name in zip([agents[index]], [envs[index]], [agent_names[index]]):
    print('Start Agent: ', agent_name)
    _, all_step_times = agent.train(env, max_episodes=max_episodes, max_steps=max_steps)
    statistics[agent_name] = {'steps total': len(all_step_times),
                              'avg step time': np.mean(all_step_times),
                              'total time': sum(all_step_times)}
print('\n')
table = PrettyTable(['Agent', 'steps total', 'avg step time', 'total time'])
for name_of_agent, data_agent in statistics.items():
    table.add_row([name_of_agent,
                   data_agent['steps total'],
                   np.round(data_agent['avg step time'], 5),
                   np.round(data_agent['total time'], 2)])
print(table)
