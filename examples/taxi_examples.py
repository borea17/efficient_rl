from efficient_rl.environment import TaxiEnvironment
from efficient_rl.agents import Rmax, FactoredRmax, DOORmax
import numpy as np


# example for self written TaxiEnvironment
grid_size = 5  # or 10
max_episodes = 5000
max_steps = 100

################ STANDARD RMAX ################
print('Rmax Agent')
classical_taxi = TaxiEnvironment(grid_size=grid_size, mode='classical MDP')
Rmax_agent = Rmax(M=1, nS=classical_taxi.nS, nA=classical_taxi.nA, gamma=0.95,
                  r_max=classical_taxi.r_max, delta=0.01, env_name='Taxi')
all_rewards, step_times = Rmax_agent.train(classical_taxi, max_episodes, max_steps)
print('steps total:{}, step time:{}, total time:{}'.format(len(step_times), np.mean(step_times),
                                                           sum(step_times)))

################ FACTORED RMAX ################
print('FactoredRmax Agent')
factored_taxi = TaxiEnvironment(grid_size=grid_size, mode='factored MDP')
FactoredRmax_agent = FactoredRmax(M=1, nS_per_var=factored_taxi.num_states_per_var,
                                  nA=factored_taxi.nA, gamma=0.95, r_max=factored_taxi.r_max,
                                  delta=0.01, DBNs=factored_taxi.DBNs, env_name='Taxi',
                                  factored_mdp_dict=factored_taxi.factored_mdp_dict)
all_rewards, step_times = FactoredRmax_agent.train(factored_taxi, max_episodes, max_steps)
print('steps total:{}, step time:{}, total time:{}'.format(len(step_times), np.mean(step_times),
                                                           sum(step_times)))

################### DOORMAX ###################
print('DOORmax Agent')
oo_taxi = TaxiEnvironment(grid_size=grid_size, mode='OO MDP')
DOORmax_agent = DOORmax(nS=oo_taxi.nS, nA=oo_taxi.nA, r_max=oo_taxi.r_max, gamma=0.95, delta=0.01,
                        k=3, num_atts=oo_taxi.num_atts, oo_mdp_dict=oo_taxi.oo_mdp_dict,
                        eff_types=['multiplication', 'assignment', 'addition'], env_name='Taxi')
all_rewards, step_times = DOORmax_agent.train(oo_taxi, max_episodes, max_steps)
print('steps total:{}, step time:{}, total time:{}'.format(len(step_times), np.mean(step_times),
                                                           sum(step_times)))
