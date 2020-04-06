import numpy as np
from prettytable import PrettyTable
from efficient_rl.agents import Rmax, FactoredRmax, QLearning
from efficient_rl.environment.classical_mdp import ClassicalTaxi
from efficient_rl.environment.factored_mdp import FactoredTaxi


max_episodes = 5000
max_steps = 100

agent_names = ['Rmax', 'Factored Rmax', 'Q Learning', 'Q Learning optimistic initalization']
envs = [ClassicalTaxi(), FactoredTaxi(), ClassicalTaxi(), ClassicalTaxi()]
agents = [Rmax(M=1, num_states=500, num_actions=6, gamma=0.95, r_max=20, delta=0.01,
               env_name='gym-Taxi'),
          FactoredRmax(M=1, num_states_per_var=[5, 5, 5, 4], num_actions=6, gamma=0.95,
                       r_max=20, delta=0.01, DBNs=envs[1].DBNs, env_name='gym-Taxi'),
          QLearning(num_states=500, num_actions=6, gamma=0.95, alpha=0.1, epsilon=0.6,
                    optimistic_init=False, env_name='gym-Taxi'),  # alpha/epsilon p.33/34 Diuks Diss
          QLearning(num_states=500, num_actions=6, gamma=0.95, alpha=1, epsilon=0, r_max=20,
                    optimistic_init=True, env_name='gym-Taxi')]  # alpha/epsilon p.33/34 Diuks Diss


statistics = {}

index = 1
for agent, env, agent_name in zip([agents[index]], [envs[index]], [agent_names[index]]):
# for agent, env, agent_name in zip(agents, envs, agent_names):
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
