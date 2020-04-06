import gym
import numpy as np
from itertools import product


class OOMDPTaxi:
    """
    taxi environment represented as an propositional OO-MDP,
    extending gym taxi environment
    """

    CLASSES = ['Wall', 'Taxi', 'Passenger', 'Destination']
    ACTION_MAPPING = {0: 'South', 1: 'North', 2: 'East', 3: 'West', 4: 'Pickup', 5: 'Dropoff'}
    # Terms is the union of all relations and additional boolean functions (passenger.in_taxi)
    TERMS = {'touch_n(taxi, wall)': None, 'touch_s(taxi, wall)': None, 'touch_e(taxi, wall)': None,
             'touch_w(taxi, wall)': None, 'on(taxi, passenger)': None,
             'on(taxi, destination)': None, 'passenger.in_taxi': None}
    # T: union of all terms plus their negations
    T = {'touch_n(taxi, wall)': None, 'not touch_n(taxi, wall)': None,
         'touch_s(taxi, wall)': None, 'not touch_s(taxi, wall)': None,
         'touch_e(taxi, wall)': None, 'not touch_e(taxi, wall)': None,
         'touch_w(taxi, wall)': None, 'not touch_w(taxi, wall)': None,
         'on(taxi, passenger)': None, 'not on(taxi, passenger)': None,
         'on(taxi, destination)': None, 'not on(taxi, destination)': None,
         'passenger.in_taxi': None, 'not passenger.in_taxi': None}

    def __init__(self):
        self.env = gym.make('Taxi-v3').env
        self.northern_border_states = OOMDPTaxi.state_list([0], range(5), range(5), range(4))
        self.southern_border_states = OOMDPTaxi.state_list([4], range(5), range(5), range(4))
        # eastern borders appear also at 6 different locations
        self.eastern_border_states = OOMDPTaxi.state_list(range(5), [4], range(5), range(4)) + \
            OOMDPTaxi.state_list([0], [1], range(5), range(4)) + \
            OOMDPTaxi.state_list([1], [1], range(5), range(4)) + \
            OOMDPTaxi.state_list([3], [0], range(5), range(4)) + \
            OOMDPTaxi.state_list([4], [0], range(5), range(4)) + \
            OOMDPTaxi.state_list([3], [2], range(5), range(4)) + \
            OOMDPTaxi.state_list([4], [2], range(5), range(4))
        # western border states appear also at 6 different locations
        self.western_border_states = OOMDPTaxi.state_list(range(5), [0], range(5), range(4)) + \
            OOMDPTaxi.state_list([0], [2], range(5), range(4)) + \
            OOMDPTaxi.state_list([1], [2], range(5), range(4)) + \
            OOMDPTaxi.state_list([3], [1], range(5), range(4)) + \
            OOMDPTaxi.state_list([4], [1], range(5), range(4)) + \
            OOMDPTaxi.state_list([3], [3], range(5), range(4)) + \
            OOMDPTaxi.state_list([4], [3], range(5), range(4))
        # taxi on passenger when taxi location = passenger location
        self.taxi_on_passenger_states = OOMDPTaxi.state_list([0], [0], [0], range(4)) + \
            OOMDPTaxi.state_list([0], [4], [1], range(4)) + \
            OOMDPTaxi.state_list([4], [0], [2], range(4)) + \
            OOMDPTaxi.state_list([4], [3], [3], range(4))
        # taxi on destination when taxi location = destination location
        self.taxi_on_destination_states = OOMDPTaxi.state_list([0], [0], range(5), [0]) + \
            OOMDPTaxi.state_list([0], [4], range(5), [1]) + \
            OOMDPTaxi.state_list([4], [0], range(5), [2]) + \
            OOMDPTaxi.state_list([4], [3], range(5), [3])
        self.passenger_in_taxi_states = OOMDPTaxi.state_list(range(5), range(5), [4], range(4))
        self.H_hat = self.compute_H_hat()
        return

    def compute_H_hat(self):
        """
            initial set of all possible hypotheses, i.e.,
            2^len(cond) * 2 => all possible conditions times 2 for predicting once True, once False
        """
        H_hat = dict()
        H_hat['predictions'] = np.array(2**7 * [True] + 2**7 * [False])
        H_hat['conditions'] = np.array(2 * list(product([0, 1], repeat=7)))
        return H_hat

    def step(self, action):
        _, reward, done, info = self.env.step(action)
        new_state = self.convert_gym_state_into_OOMDP_state()
        self.s = new_state
        return new_state, reward, done, info

    def cond(self):
        """
            returns a list where each entry defines whether the corresponding
            entry in OOMDPTaxi.TERMS is True (1) or False (0)
        """
        state = self.env.s
        out_list = 7 * [None]
        out_list[0] = state in self.northern_border_states
        out_list[1] = state in self.southern_border_states
        out_list[2] = state in self.eastern_border_states
        out_list[3] = state in self.western_border_states
        out_list[4] = state in self.taxi_on_passenger_states
        out_list[5] = state in self.taxi_on_destination_states
        out_list[6] = state in self.passenger_in_taxi_states
        return out_list

    def eff(state, new_state, att):
        """
        this function returns one effect of each type that would transform attribute att in state
        s into it s value in new state s'
        """
        pass

    def convert_gym_state_into_OOMDP_state(self):
        """
            state is the union of the attribute values of all objects,
            for simplicity, we only take the states that can change
        """
        taxi_y, taxi_x, pass_loc_i, dest_loc_i = list(self.env.decode(self.env.s))
        if pass_loc_i == 4:
            pass_y, pass_x, in_taxi = taxi_y, taxi_x, True
        else:
            pass_y, pass_x = self.env.locs[pass_loc_i]
            in_taxi = False
        dest_y, dest_x = self.env.locs[dest_loc_i]

        state = [taxi_x, taxi_y, in_taxi, pass_x, pass_y]
        return state

    def reset(self):
        self.env.reset()
        self.s = self.convert_gym_state_into_OOMDP_state()
        return

    def render(self):
        self.env.render()
        return

    def set_state(self, state):
        taxi_row, taxi_col, pass_loc, dest_idx = state[0], state[1], state[2], state[3]
        self.env.s = self.env.encode(taxi_row, taxi_col, pass_loc, dest_idx)
        self.s = self.convert_gym_state_into_OOMDP_state()
        return

    @staticmethod
    def state_list(taxi_y_it, taxi_x_it, pass_loc_i_it, dest_loc_i_it):
        """
            generation of state lists for later use in function cond
        """
        helper_gym = gym.make('Taxi-v3').env

        state_list = []
        for taxi_y in taxi_y_it:
            for taxi_x in taxi_x_it:
                for pass_loc_i in pass_loc_i_it:
                    for dest_loc_i in dest_loc_i_it:
                        state = helper_gym.encode(taxi_y, taxi_x, pass_loc_i, dest_loc_i)
                        state_list.append(state)
        return state_list

    def current_terms(self):
        """
            returns the subset of terms in T that are true in the current state,
            conversion cond(s) into human readable format
        """
        state = self.env.s
        cond_string = self.cond(state)
        cur_cond = []
        cur_cond.append('touch_n(taxi, wall)' if cond_string[0] else 'not touch_n(taxi, wall)')
        cur_cond.append('touch_s(taxi, wall)' if cond_string[1] else 'not touch_s(taxi, wall)')
        cur_cond.append('touch_e(taxi, wall)' if cond_string[2] else 'not touch_e(taxi, wall)')
        cur_cond.append('touch_w(taxi, wall)' if cond_string[3] else 'not touch_w(taxi, wall)')
        cur_cond.append('on(taxi, passenger)' if cond_string[4] else 'not on(taxi, passenger)')
        cur_cond.append('on(taxi, destination)' if cond_string[5] else 'not on(taxi, destination)')
        cur_cond.append('passenger.in-taxi=T' if cond_string[6] else 'passenger.in-taxi=F')
        return cur_cond

    def human_play(self):
        self.reset()
        state = self.s
        while True:
            self.env.render()
            print(state)
            print(self.cond())
            action = input()
            new_state, reward, done, _ = self.step(int(action))
            if done:
                break
            state = new_state
        return


if __name__ == '__main__':
    OOMDP_env = OOMDPTaxi()
    OOMDP_env.human_play()
