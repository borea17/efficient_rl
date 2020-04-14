import gym
import numpy as np


class OOTaxi:
    """
    taxi environment represented as an propositional OO MDP,
    extending gym taxi environment
    """

    # Terms is the union of all relations and additional boolean functions (passenger.in_taxi)
    TERMS = ['touch_n(taxi, wall)', 'touch_s(taxi, wall)', 'touch_e(taxi, wall)',
             'touch_w(taxi, wall)', 'on(taxi, passenger)',
             'on(taxi, destination)', 'passenger.in_taxi']

    def __init__(self):
        self.env = gym.make('Taxi-v3').env
        self.northern_border_states = OOTaxi.state_list([0], range(5), range(5), range(4))
        self.southern_border_states = OOTaxi.state_list([4], range(5), range(5), range(4))
        # eastern borders appear also at 6 different locations
        self.eastern_border_states = OOTaxi.state_list(range(5), [4], range(5), range(4)) + \
            OOTaxi.state_list([0], [1], range(5), range(4)) + \
            OOTaxi.state_list([1], [1], range(5), range(4)) + \
            OOTaxi.state_list([3], [0], range(5), range(4)) + \
            OOTaxi.state_list([4], [0], range(5), range(4)) + \
            OOTaxi.state_list([3], [2], range(5), range(4)) + \
            OOTaxi.state_list([4], [2], range(5), range(4))
        # western border states appear also at 6 different locations
        self.western_border_states = OOTaxi.state_list(range(5), [0], range(5), range(4)) + \
            OOTaxi.state_list([0], [2], range(5), range(4)) + \
            OOTaxi.state_list([1], [2], range(5), range(4)) + \
            OOTaxi.state_list([3], [1], range(5), range(4)) + \
            OOTaxi.state_list([4], [1], range(5), range(4)) + \
            OOTaxi.state_list([3], [3], range(5), range(4)) + \
            OOTaxi.state_list([4], [3], range(5), range(4))
        self.passenger_in_taxi_states = OOTaxi.state_list(range(5), range(5), [4], range(4))

        self.num_atts = 7  # see state in convert_gym_state_into_OO_MDP_state()
        self.oo_mdp_dict = self.create_oo_mdp_state_dict()
        return

    def step(self, action):
        """
            step is slightly different to gym as dropoff is only possible at destination
        """
        if action == 5:  # drop off action
            taxi_row, taxi_col, pass_loc, dest_loc = list(self.env.decode(self.env.s))
            if (taxi_row, taxi_col) in self.env.locs:  # taxi location on any predefined locations
                if self.env.locs.index((taxi_row, taxi_col)) != dest_loc:
                    # illegal drop off action following Diuk
                    new_state = self.convert_gym_state_into_OO_MDP_state()
                    reward = -10
                    done = False
                    info = None
                    self.env.lastaction = 5
                    return new_state, reward, done, info
            _, reward, done, info = self.env.step(action)
        else:
            _, reward, done, info = self.env.step(action)
        new_state = self.convert_gym_state_into_OO_MDP_state()
        self.s = new_state
        return new_state, reward, done, info

    def cond(self):
        """
            returns an array with length of TERMS where each entry defines whether the corresponding
            entry in OOMDPTaxi.TERMS is True (1) or False (0)
        """
        classical_state = self.env.s
        taxi_y, taxi_x, pass_loc_i, dest_loc_i = list(self.env.decode(classical_state))

        out_array = np.zeros(len(OOTaxi.TERMS))

        out_array[0] = classical_state in self.northern_border_states
        out_array[1] = classical_state in self.southern_border_states
        out_array[2] = classical_state in self.eastern_border_states
        out_array[3] = classical_state in self.western_border_states
        out_array[4] = self.taxi_x == self.pass_x and self.taxi_y == self.pass_y
        out_array[5] = self.taxi_x == self.dest_x and self.taxi_y == self.dest_y
        out_array[6] = classical_state in self.passenger_in_taxi_states
        return out_array

    def convert_gym_state_into_OO_MDP_state(self):
        """
            for convenience a tuple (state, condition) is returned here,
            this allows for a generic definition of the RmaxBaseClass
        """
        taxi_y, taxi_x, pass_loc_i, dest_loc_i = list(self.env.decode(self.env.s))
        self.taxi_x, self.taxi_y = taxi_x, taxi_y

        if pass_loc_i == 4:
            pass_y, pass_x, in_taxi = self.pass_y, self.pass_x, True
        else:
            pass_y, pass_x = self.env.locs[pass_loc_i]
            self.pass_x, self.pass_y = pass_x, pass_y
            in_taxi = False

        dest_y, dest_x = self.env.locs[dest_loc_i]
        self.dest_x, self.dest_y = dest_x, dest_y

        state = [taxi_x, taxi_y, pass_x, pass_y, dest_x, dest_y, int(in_taxi)]
        self.condition = self.cond()
        return (state, self.condition)

    def reset(self):
        """
            in gym reset, passenger location and destination location are never the same,
            in original of Diettrich this is possible
        """
        taxi_row, taxi_colum = np.random.randint(5), np.random.randint(5)
        pass_loc = np.random.randint(4)
        dest_loc = np.random.randint(4)
        self.env.s = self.env.encode(taxi_row, taxi_colum, pass_loc, dest_loc)
        self.s = self.convert_gym_state_into_OO_MDP_state()
        return self.s

    def render(self):
        self.env.render()
        return

    def encode(self, taxi_row, taxi_col, pass_loc, dest_idx, pass_loc_in_taxi=None):
        if pass_loc == len(self.env.locs):
            self.pass_y, self.pass_x = self.env.locs[pass_loc_in_taxi]

        self.env.s = self.env.encode(taxi_row, taxi_col, pass_loc, dest_idx)
        self.s = self.convert_gym_state_into_OO_MDP_state()
        return self.s

    def create_oo_mdp_state_dict(self):
        """
            create oo_mdp_state-condition to flat_state and vice versa mapping using a dictionary,
            NOTE: - different oo_mdp_states map to the same flat_state
        """
        oo_mdp_dict = dict()
        oo_mdp_dict['oo_mdp_to_flat_map'] = dict()
        oo_mdp_dict['flat_to_oo_mdp_map'] = [[] for flat_state in range(self.env.nS)]

        i_pass_in_taxi = len(self.env.locs)

        for taxi_y in range(5):
            for taxi_x in range(5):
                for idx_pass in range(len(self.env.locs)):
                    for idx_dest in range(len(self.env.locs)):
                        for in_taxi in [False, True]:
                            if in_taxi:
                                # all combinations of passenger locations if passenger in taxi
                                state_cond = self.encode(taxi_y, taxi_x, i_pass_in_taxi, idx_dest,
                                                         idx_pass)
                            else:
                                state_cond = self.encode(taxi_y, taxi_x, idx_pass, idx_dest)

                            oo_mdp_s_tuple = tuple(state_cond[0])
                            flat_state = self.env.s

                            oo_mdp_dict['oo_mdp_to_flat_map'][oo_mdp_s_tuple] = flat_state
                            oo_mdp_dict['flat_to_oo_mdp_map'][flat_state].append(state_cond)
        return oo_mdp_dict

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
    OO_MDP_env = OOTaxi()
    OO_MDP_env.human_play()
