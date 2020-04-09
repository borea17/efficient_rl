from unittest import TestCase
from efficient_rl.environment import TaxiEnvironment
from efficient_rl.environment.classical_mdp import ClassicalTaxi
from efficient_rl.environment.factored_mdp import FactoredTaxi
from efficient_rl.environment.oo_mdp import OOTaxi
import numpy as np


class testTaxiEnvironment(TestCase):

    max_steps = 1000

    def setUp(self):
        self.classical_gym_env = ClassicalTaxi()
        self.classical_taxi_env = TaxiEnvironment(grid_size=5, mode='classical MDP')
        self.factored_gym_env = FactoredTaxi()
        self.factored_taxi_env = TaxiEnvironment(grid_size=5, mode='factored MDP')
        self.oo_gym_env = OOTaxi()
        self.oo_taxi_env = TaxiEnvironment(grid_size=5, mode='OO MDP')
        return

    def test_oo_mdp_gym_env_and_oo_mdp_taxi_env_do_the_same(self):
        gym_env = self.oo_gym_env
        taxi_env = self.oo_taxi_env

        # check for equality of dictionaries skipped since predefined locations differ due
        # different y axis orientation

        (gym_s, gym_condition) = gym_env.reset()
        gym_row, gym_col, gym_pass, gym_dest = list(gym_env.env.decode(gym_env.env.s))
        # only y axis (gym row) is inverted in relation to taxi env
        taxi_row = (4 - gym_row) % 5
        (taxi_s, taxi_condition) = taxi_env.set_state(taxi_row, gym_col, gym_pass, gym_dest)

        # transform y coordinates of gym_s
        gym_s[1] = (4 - gym_s[1]) % 5
        gym_s[3] = (4 - gym_s[3]) % 5
        gym_s[5] = (4 - gym_s[5]) % 5
        self.assertTrue(np.all(gym_s == taxi_s))
        self.assertTrue(np.all(gym_condition == taxi_condition))

        for _ in range(testTaxiEnvironment.max_steps):
            action = np.random.randint(6)

            new_state_gym, reward_gym, done_gym, _ = gym_env.step(action)
            new_state_taxi, reward_taxi, done_taxi, _ = taxi_env.step(action)

            oo_mdp_state_gym, condition_gym = new_state_gym[0], new_state_gym[1]
            oo_mdp_state_taxi, condition_taxi = new_state_taxi[0], new_state_taxi[1]
            # transform y coordinates of oo mdp gym state
            oo_mdp_state_gym[1] = (4 - oo_mdp_state_gym[1]) % 5
            oo_mdp_state_gym[3] = (4 - oo_mdp_state_gym[3]) % 5
            oo_mdp_state_gym[5] = (4 - oo_mdp_state_gym[5]) % 5

            self.assertTrue(np.all(oo_mdp_state_gym == oo_mdp_state_taxi))
            self.assertTrue(np.all(condition_gym == condition_taxi))
            self.assertTrue(reward_gym == reward_taxi)
            self.assertTrue(done_gym == done_taxi)

            if done_gym:
                break
        return

    def test_factored_gym_env_and_factored_taxi_env_do_the_same(self):
        gym_env = self.factored_gym_env
        taxi_env = self.factored_taxi_env

        # check for equality of dictionaries
        self.assertTrue(gym_env.factored_mdp_dict == taxi_env.factored_mdp_dict)

        gym_state = gym_env.reset()
        gym_row, gym_col, gym_pass, gym_dest = list(gym_env.env.decode(gym_env.env.s))
        # only y axis (gym row) is inverted in relation to taxi env
        taxi_row = (4 - gym_row) % 5
        taxi_state = taxi_env.set_state(taxi_row, gym_col, gym_pass, gym_dest)

        # transform y coordinates
        gym_state[0] = (4 - gym_state[0]) % 5
        self.assertTrue(np.all(gym_state == taxi_state))

        for _ in range(testTaxiEnvironment.max_steps):
            action = np.random.randint(6)

            new_state_gym, reward_gym, done_gym, _ = gym_env.step(action)
            new_state_taxi, reward_taxi, done_taxi, _ = taxi_env.step(action)

            # transform y coordinates
            new_state_gym[0] = (4 - new_state_gym[0]) % 5

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
            if gym_pass != taxi_pass:
                print(gym_pass, taxi_pass)
            self.assertTrue(gym_pass == taxi_pass)
            self.assertTrue(gym_dest == taxi_dest)

            self.assertTrue(reward_gym == reward_taxi)
            self.assertTrue(done_gym == done_taxi)

            if done_gym:
                break
        return
