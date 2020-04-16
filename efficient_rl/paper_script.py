from prettytable import PrettyTable
from efficient_rl.agents import Rmax, FactoredRmax, DOORmax, QLearning
from efficient_rl.environment import TaxiEnvironment
import numpy as np

# setup
n_repetitions = 100
max_episodes = 10000
max_steps = 100
agents_to_compare = [
    'Q Learning - optimistic initialization',
    'Rmax',
    'Factored Rmax',
    'DOORmax'
]


def compare_agent(agent_name):
    if agent_name in ['Q Learning - optimistic initialization', 'Rmax']:
        envs = [TaxiEnvironment(grid_size=5, mode='classical MDP'),
                TaxiEnvironment(grid_size=10, mode='classical MDP')]
    elif agent_name == 'Factored Rmax':
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
        if agent_name == 'Q Learning - optimistic initialization':
            agent = QLearning(nS=env.nS, nA=env.nA, r_max=env.r_max, gamma=0.95, alpha=1, epsilon=0,
                              optimistic_init=True, env_name='Taxi')
        elif agent_name == 'Rmax':
            agent = Rmax(M=1, nS=env.nS, nA=env.nA, r_max=env.r_max, gamma=0.95, delta=0.01,
                         env_name='Taxi')
        elif agent_name == 'Factored Rmax':
            agent = FactoredRmax(M=1, nS_per_var=env.num_states_per_var, nA=env.nA, r_max=env.r_max,
                                 gamma=0.95, delta=0.01, DBNs=env.DBNs,
                                 factored_mdp_dict=env.factored_mdp_dict, env_name='Taxi')
        elif agent_name == 'DOORmax':
            agent = DOORmax(nS=env.nS, nA=env.nA, r_max=env.r_max, gamma=0.95, delta=0.01, k=3,
                            num_atts=env.num_atts, oo_mdp_dict=env.oo_mdp_dict, env_name='Taxi',
                            eff_types=['assignment', 'addition', 'multiplication'])

        print(' start with', env_name)
        all_step_times = []
        for i_rep in range(n_repetitions):
            print('   current repetition: ', i_rep + 1, '/', n_repetitions)
            _, step_times = agent.train(env, max_episodes=max_episodes, max_steps=max_steps,
                                        show_intermediate=False)
            all_step_times.extend(step_times)

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
