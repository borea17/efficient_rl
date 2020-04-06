from unittest import TestCase
from OOMDP_Taxi.environment import TaxiEnvironment
from OOMDP_Taxi.environment.classical_mdp import ClassicalTaxi
from OOMDP_Taxi.environment.factored_mdp import FactoredTaxi
import numpy as np


class testTaxiEnvironment(TestCase):

    max_steps = 1000

    def setUp(self):
        self.classical_gym_env = ClassicalTaxi()
        self.classical_taxi_env = TaxiEnvironment(grid_size=5, mode='classical MDP')
        self.factored_gym_env = FactoredTaxi()
        self.factored_taxi_env = TaxiEnvironment(grid_size=5, mode='factored MDP')
        return

    def test_factored_gym_env_and_factored_taxi_env_do_the_same(self):
        gym_env = self.factored_gym_env
        taxi_env = self.factored_taxi_env

        gym_row, gym_col, gym_pass, gym_dest = gym_env.reset()
        # only y axis (gym row) in relation to taxi env
        taxi_row = (4 - gym_row) % 5
        taxi_state = taxi_env.set_state(taxi_row, gym_col, gym_pass, gym_dest)
        for _ in range(testTaxiEnvironment.max_steps):
            action = np.random.randint(6)

            new_state_gym, reward_gym, done_gym, _ = gym_env.step(action)
            new_state_taxi, reward_taxi, done_taxi, _ = taxi_env.step(action)

            new_state_gym[0] = (4 - new_state_gym[0]) % 5
            if not np.all(new_state_gym == new_state_taxi):
                print(new_state_gym, new_state_taxi)
            self.assertTrue(np.all(new_state_gym == new_state_taxi))
            self.assertTrue(reward_gym == reward_taxi)
            self.assertTrue(done_gym == done_taxi)

            if done_gym:
                break
        return

    def test_classical_gym_env_and_classical_taxi_env_do_the_same(self):
        gym_env = self.classical_gym_env
        taxi_env = self.classical_taxi_env

        gym_state = gym_env.reset()
        gym_row, gym_col, gym_pass, gym_dest = list(gym_env.decode(gym_state))
        # only y axis (gym row) is inverted in relation to taxi env
        taxi_row = (4 - gym_row) % 5
        taxi_state = taxi_env.set_state(taxi_row, gym_col, gym_pass, gym_dest)
        for _ in range(testTaxiEnvironment.max_steps):
            action = np.random.randint(6)
            new_state_gym, reward_gym, done_gym, _ = gym_env.step(action)
            new_state_taxi, reward_taxi, done_taxi, _ = taxi_env.step(action)

            gym_row, gym_col, gym_pass, gym_dest = list(gym_env.decode(new_state_gym))
            gym_row_in_taxi_env = (4 - gym_row) % 5
            taxi_row, taxi_col, taxi_pass, taxi_dest = list(gym_env.decode(new_state_taxi))

            self.assertTrue(gym_row_in_taxi_env == taxi_row)
            self.assertTrue(gym_col == taxi_col)
            self.assertTrue(gym_pass == taxi_pass)
            self.assertTrue(gym_dest == taxi_dest)

            self.assertTrue(reward_gym == reward_taxi)
            self.assertTrue(done_gym == done_taxi)

            if done_gym:
                break
        return
