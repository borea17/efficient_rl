from prettytable import PrettyTable
from efficient_rl.agents import FactoredRmax, DOORmax
from efficient_rl.environment import TaxiEnvironment
import numpy as np

# setup
n_repetitions = 1
max_episodes = 100000
max_steps = 100
agents_to_compare = [
   'Factored Rmax',
   'DOORmax'
]


def compare_agent(agent_name):
    if agent_name == 'Factored Rmax':
        envs = [TaxiEnvironment(grid_size=5, mode='factored MDP'),
                TaxiEnvironment(grid_size=10, mode='factored MDP')]
    elif agent_name == 'DOORmax':
        envs = [TaxiEnvironment(grid_size=5, mode='OO MDP'),
                TaxiEnvironment(grid_size=10, mode='OO MDP')]
    else:
        raise NameError('agent name unknown')

    print('Start agent: ', agent_name)

    current_statistics = {}
    for env, env_name in zip(envs, ['Taxi 5x5', 'Taxi 10x10']):
        if agent_name == 'Factored Rmax':
            agent = FactoredRmax(M=1, nS_per_var=env.num_states_per_var, nA=env.nA, r_max=env.r_max,
                                 gamma=0.95, delta=0.1, DBNs=env.DBNs,
                                 factored_mdp_dict=env.factored_mdp_dict, env_name='Taxi')
        elif agent_name == 'DOORmax':
            agent = DOORmax(nS=env.nS, nA=env.nA, r_max=env.r_max, gamma=0.95, delta=0.1, k=5,
                            num_atts=env.num_atts, oo_mdp_dict=env.oo_mdp_dict, env_name='Taxi',
                            eff_types=['assignment', 'addition'])

        print(' start with', env_name)
        all_step_times = []
        for i_rep in range(n_repetitions):
            print('   current repetition: ', i_rep + 1, '/', n_repetitions)
            _, step_times = agent.train(env, max_episodes=max_episodes, max_steps=max_steps)

            agent.reset()

            all_step_times.extend(step_times)
            print(' current avg # steps', len(all_step_times)/(i_rep + 1))
            print(' current avg Time/step', np.mean(all_step_times))

        print(' avg # steps', len(all_step_times)/n_repetitions)
        print(' avg Time/step', np.mean(all_step_times))

        current_statistics[env_name + ' #steps'] = len(all_step_times)/n_repetitions
        current_statistics[env_name + ' Time/step'] = np.mean(all_step_times)
    return current_statistics


statistics = {}
for agent_name in agents_to_compare:
    statistics[agent_name] = compare_agent(agent_name)

print('\n Paper Results \n')
table = PrettyTable(['Agent', 'Taxi 5x5 #steps', 'Taxi 5x5 Time/step',
                     'Taxi 10x10 #steps', 'Taxi 10x10 Time/step'])
for name_of_agent, data_agent in statistics.items():
    print(data_agent)
    table.add_row([name_of_agent,
                   data_agent['Taxi 5x5 #steps'],
                   np.round(data_agent['Taxi 5x5 Time/step'], 5),
                   data_agent['Taxi 10x10 #steps'],
                   np.round(data_agent['Taxi 10x10 Time/step'], 5)])
print(table)
