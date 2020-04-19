import numpy as np
from efficient_rl.oo_mdp_operations import Operations as oo_mdp_OP
from efficient_rl.environment.oo_mdp import TaxiRelations as Rel
from gym.utils import seeding, colorize


class TaxiEnvironment:

    """
        taxi environment represented as a propositional OO MDP,
        which can also be used for classical and factored representation

        this class shall demonstrate in more detail how the enviroment interaction
        is understood in an propositional OOMDP

        for simplicity, a wall_list is created where x, y, position are the corresponding
        wall attributes instead of creating several wall instances

        it can also be used to create different versions of the TAXI domain
    """

    ACTION_MAPPING = {0: 'South', 1: 'North', 2: 'East', 3: 'West', 4: 'Pickup', 5: 'Dropoff'}
    CLASSES = {'taxi': ['x', 'y'],
               'passenger': ['x', 'y', 'in_taxi'],
               'destination': ['x', 'y'],
               'wall_list': ['x', 'y', 'position']}
    # Terms is the union of all relations and additional boolean functions (passenger.in_taxi)
    TERMS = ['touch_n(taxi, wall)', 'touch_s(taxi, wall)', 'touch_e(taxi, wall)',
             'touch_w(taxi, wall)', 'on(taxi, passenger)',
             'on(taxi, destination)', 'passenger.in_taxi']

    def __init__(self, grid_size=5, mode='classical MDP'):
        assert mode in ['classical MDP', 'factored MDP', 'OO MDP']
        self.mode = mode
        self.grid_size = grid_size
        # predefined locations of passenger and destination
        if grid_size == 5:
            self.PREDEFINED_LOCATIONS = [(0, 4), (4, 4), (0, 0), (3, 0)]
            self.POSITION_NAMES = ['R', 'G', 'Y', 'B']
        elif grid_size == 10:
            self.PREDEFINED_LOCATIONS = [(0, 1), (0, 9), (5, 9), (4, 0),
                                         (3, 6), (6, 5), (8, 9), (9, 0)]
            self.POSITION_NAMES = ['Y', 'R', 'G', 'B', 'W', 'M', 'C', 'P']
        else:
            raise NotImplementedError()
        # objects initialization
        objs = {}
        for class_name in TaxiEnvironment.CLASSES:
            current_att_dict = dict()
            for att in TaxiEnvironment.CLASSES[class_name]:
                current_att_dict[att] = None
            objs[class_name] = current_att_dict
        self.objs = objs

        self.define_action_conditions()

        n_predefined_locs = len(self.PREDEFINED_LOCATIONS)
        # num_states for: taxi_x, taxi_y, passenger_loc (+1: in_taxi), destination_loc
        self.num_states_per_var = [grid_size, grid_size, n_predefined_locs + 1, n_predefined_locs]
        self.nS, self.nA = np.prod(self.num_states_per_var), 6
        self.r_max = 20
        self.x_wall, self.y_wall, self.position_wall = self.get_wall_coordinates_and_positions()

        if mode == 'classical MDP':
            self.make_state = self.make_classical_MDP_state
        elif mode == 'factored MDP':
            from efficient_rl.environment.factored_mdp import FactoredTaxi

            self.make_state = self.make_factored_MDP_state
            self.DBNs = FactoredTaxi.create_DBNs()
            self.factored_mdp_dict = self.create_factored_mdp_state_dict()
        elif mode == 'OO MDP':
            self.make_state = self.make_OO_MDP_state
            self.num_atts = 7  # == len(state) see make_OO_MDP_state
            self.oo_mdp_dict = self.create_oo_mdp_state_dict()
        else:
            raise NotImplementedError
        return

    def cond(self):
        """
            returns an array with length of TERMS where each entry defines whether the corresponding
            entry in OOMDPTaxi.TERMS is True (1) or False (0)
        """
        out_array = np.zeros([len(TaxiEnvironment.TERMS)])

        out_array[0] = Rel.touch_north(self.objs['taxi'], self.objs['wall_list'])
        out_array[1] = Rel.touch_south(self.objs['taxi'], self.objs['wall_list'])
        out_array[2] = Rel.touch_east(self.objs['taxi'], self.objs['wall_list'])
        out_array[3] = Rel.touch_west(self.objs['taxi'], self.objs['wall_list'])
        out_array[4] = Rel.on(self.objs['taxi'], self.objs['passenger'])
        out_array[5] = Rel.on(self.objs['taxi'], self.objs['destination'])
        out_array[6] = self.objs['passenger']['in_taxi']

        return out_array

    def step(self, action_int):
        action = TaxiEnvironment.ACTION_MAPPING[action_int]
        reward, done = -1, False

        if self.mode == 'OO MDP':
            current_condition = self.condition
        else:
            current_condition = self.cond()

        if action == 'North':
            if oo_mdp_OP.hypothesis_matches(self.north_transition_condition, current_condition):
                self.objs['taxi']['y'] += 1
        elif action == 'South':
            if oo_mdp_OP.hypothesis_matches(self.south_transition_condition, current_condition):
                self.objs['taxi']['y'] -= 1
        elif action == 'East':
            if oo_mdp_OP.hypothesis_matches(self.east_transition_condition, current_condition):
                self.objs['taxi']['x'] += 1
        elif action == 'West':
            if oo_mdp_OP.hypothesis_matches(self.west_transition_condition, current_condition):
                self.objs['taxi']['x'] -= 1
        elif action == 'Pickup':
            if oo_mdp_OP.hypothesis_matches(self.pick_up_transition_condition, current_condition):
                self.objs['passenger']['in_taxi'] = True
            else:
                reward = -10
        elif action == 'Dropoff':
            if oo_mdp_OP.hypothesis_matches(self.drop_off_transition_condition, current_condition):
                self.objs['passenger']['in_taxi'] = False
                if self.mode in ['classical MDP', 'factored MDP']:
                    self.objs['passenger']['x'] = self.objs['taxi']['x']
                    self.objs['passenger']['y'] = self.objs['taxi']['y']
                reward, done = 20, True
            else:
                reward = -10
        self.score += reward

        new_state = self.make_state()
        self.s = new_state
        return new_state, reward, done, self.score

    def define_action_conditions(self):
        nan = np.nan
        self.north_transition_condition = np.array([False, nan, nan, nan, nan, nan, nan])
        self.south_transition_condition = np.array([nan, False, nan, nan, nan, nan, nan])
        self.east_transition_condition = np.array([nan, nan, False, nan, nan, nan, nan])
        self.west_transition_condition = np.array([nan, nan, nan, False, nan, nan, nan])
        self.pick_up_transition_condition = np.array([nan, nan, nan, nan, True, nan, False])
        self.drop_off_transition_condition = np.array([nan, nan, nan, nan, nan, True, True])
        return

    def make_classical_MDP_state(self):
        passenger_loc = (self.objs['passenger']['x'], self.objs['passenger']['y'])
        destination_loc = (self.objs['destination']['x'], self.objs['destination']['y'])
        if self.objs['passenger']['in_taxi']:
            idx_pass = len(self.PREDEFINED_LOCATIONS)
        else:
            idx_pass = self.PREDEFINED_LOCATIONS.index(passenger_loc)
        idx_dest = self.PREDEFINED_LOCATIONS.index(destination_loc)
        state = self.encode(self.objs['taxi']['y'], self.objs['taxi']['x'], idx_pass, idx_dest)
        return state

    def make_factored_MDP_state(self):
        passenger_loc = (self.objs['passenger']['x'], self.objs['passenger']['y'])
        destination_loc = (self.objs['destination']['x'], self.objs['destination']['y'])
        if self.objs['passenger']['in_taxi']:
            idx_passenger = len(self.PREDEFINED_LOCATIONS)
        else:
            idx_passenger = self.PREDEFINED_LOCATIONS.index(passenger_loc)
        idx_destination = self.PREDEFINED_LOCATIONS.index(destination_loc)
        state = [self.objs['taxi']['y'], self.objs['taxi']['x'], idx_passenger, idx_destination]
        return state

    def make_OO_MDP_state(self):
        """
            for convenience a tuple (state, condition) is returned here,
            this allows for a generic definition of the RmaxBaseClass
        """
        taxi_x, taxi_y = self.objs['taxi']['x'], self.objs['taxi']['y']
        pass_x, pass_y = self.objs['passenger']['x'], self.objs['passenger']['y']
        dest_x, dest_y = self.objs['destination']['x'], self.objs['destination']['y']
        in_taxi = self.objs['passenger']['in_taxi']

        self.condition = self.cond()
        state = [taxi_x, taxi_y, pass_x, pass_y, dest_x, dest_y, int(in_taxi)]
        return (state, self.condition)

    def create_factored_mdp_state_dict(self):
        """
            create factored_state to flat_state and vice versa mapping using a dictionary
            NOTE: there is a one-to-one correspondence between factored and flat state, i.e.,
                  if passenger is in_taxi all factored states with different idx_pass_ad are equal
        """
        self.reset()

        factored_mdp_dict = dict()
        factored_mdp_dict['factored_to_flat_map'] = dict()
        factored_mdp_dict['flat_to_factored_map'] = [[] for flat_states in range(self.nS)]

        for taxi_y in range(self.grid_size):
            for taxi_x in range(self.grid_size):
                for idx_pass in range(len(self.PREDEFINED_LOCATIONS)):
                    for idx_dest in range(len(self.PREDEFINED_LOCATIONS)):
                        for in_taxi in [False, True]:
                            if in_taxi:
                                # all combinations of passenger locations if passenger in taxi
                                idx_pass_ad = len(self.PREDEFINED_LOCATIONS)
                                factored_s = self.set_state(taxi_y, taxi_x, idx_pass_ad, idx_dest,
                                                            idx_pass)
                            else:
                                factored_s = self.set_state(taxi_y, taxi_x, idx_pass, idx_dest)

                            factored_tup = tuple(factored_s)
                            flat_state = self.make_classical_MDP_state()

                            factored_mdp_dict['factored_to_flat_map'][factored_tup] = flat_state
                            factored_mdp_dict['flat_to_factored_map'][flat_state] = factored_s
        return factored_mdp_dict

    def create_oo_mdp_state_dict(self):
        """
            create oo_mdp_state-condition to flat_state and vice versa mapping using a dictionary,
            NOTE: - different oo_mdp_states map to the same flat_state
        """
        oo_mdp_dict = dict()
        oo_mdp_dict['oo_mdp_to_flat_map'] = dict()
        oo_mdp_dict['flat_to_oo_mdp_map'] = [[] for flat_state in range(self.nS)]

        i_pass_in_taxi = len(self.PREDEFINED_LOCATIONS)

        for taxi_y in range(self.grid_size):
            for taxi_x in range(self.grid_size):
                for idx_pass in range(len(self.PREDEFINED_LOCATIONS)):
                    for idx_dest in range(len(self.PREDEFINED_LOCATIONS)):
                        for in_taxi in [False, True]:
                            if in_taxi:
                                # all combinations of passenger locations if passenger in taxi
                                state_cond = self.set_state(taxi_y, taxi_x, i_pass_in_taxi,
                                                            idx_dest, idx_pass)
                            else:
                                state_cond = self.set_state(taxi_y, taxi_x, idx_pass, idx_dest)

                            oo_mdp_s_tuple = tuple(state_cond[0])
                            flat_state = self.make_classical_MDP_state()

                            oo_mdp_dict['oo_mdp_to_flat_map'][oo_mdp_s_tuple] = flat_state
                            oo_mdp_dict['flat_to_oo_mdp_map'][flat_state].append(state_cond)
        return oo_mdp_dict

    def reset(self):
        rng, seed = seeding.np_random()

        grid_size = self.grid_size

        n_locs = len(self.PREDEFINED_LOCATIONS)

        pass_i, dest_i = rng.randint(0, n_locs), rng.randint(0, n_locs)
        pass_loc, dest_loc = self.PREDEFINED_LOCATIONS[pass_i], self.PREDEFINED_LOCATIONS[dest_i]
        taxi_loc = (rng.randint(0, grid_size), rng.randint(0, grid_size))
        x_wall, y_wall, position_wall = self.x_wall, self.y_wall, self.position_wall

        self.objs['taxi']['x'], self.objs['taxi']['y'] = taxi_loc[0], taxi_loc[1]
        self.objs['passenger']['x'], self.objs['passenger']['y'] = pass_loc[0], pass_loc[1]
        self.objs['passenger']['in_taxi'] = False
        self.objs['destination']['x'], self.objs['destination']['y'] = dest_loc[0], dest_loc[1]
        self.objs['wall_list']['x'], self.objs['wall_list']['y'] = x_wall, y_wall
        self.objs['wall_list']['position'] = position_wall

        self.score = 0
        self.s = self.make_state()
        return self.s

    def set_state(self, taxi_y, taxi_x, pass_loc, dest_idx, pass_loc_in_taxi=None):
        if pass_loc == len(self.PREDEFINED_LOCATIONS):  # passenger in_taxi
            in_taxi = True
            pass_x, pass_y = self.PREDEFINED_LOCATIONS[pass_loc_in_taxi]
        else:
            in_taxi = False
            pass_x, pass_y = self.PREDEFINED_LOCATIONS[pass_loc]
        dest_x, dest_y = self.PREDEFINED_LOCATIONS[dest_idx]

        self.reset()
        self.objs['passenger']['in_taxi'] = in_taxi
        self.objs['taxi']['x'], self.objs['taxi']['y'] = taxi_x, taxi_y
        self.objs['passenger']['x'], self.objs['passenger']['y'] = pass_x, pass_y
        self.objs['destination']['x'], self.objs['destination']['y'] = dest_x, dest_y
        self.s = self.make_state()
        return self.s

    def encode(self, taxi_row, taxi_col, pass_loc, dest_idx):
        """
        see encode function of gym
        num_states_per_var: (grid_size), grid_size, len(self.PREDEFINED) + 1, len(self.PREDEFINED)
        """
        i = taxi_row
        i *= self.num_states_per_var[1]
        i += taxi_col
        i *= self.num_states_per_var[2]
        i += pass_loc
        i *= self.num_states_per_var[3]
        i += dest_idx
        return i

    def get_wall_coordinates_and_positions(self):
        grid_size = self.grid_size
        x_wall, y_wall, position_wall = [], [], []
        # surronding walls
        for index in range(grid_size):
            x_wall.append(0), y_wall.append(index), position_wall.append('left')
            x_wall.append(grid_size - 1), y_wall.append(index), position_wall.append('right')
            x_wall.append(index), y_wall.append(0), position_wall.append('above')
            x_wall.append(index), y_wall.append(grid_size - 1), position_wall.append('below')
        # inner walls
        if grid_size == 5:
            x_wall.append(0), y_wall.append(0), position_wall.append('right')  # 21
            x_wall.append(0), y_wall.append(1), position_wall.append('right')  # 22
            x_wall.append(1), y_wall.append(3), position_wall.append('right')  # 23
            x_wall.append(1), y_wall.append(4), position_wall.append('right')  # 24
            x_wall.append(2), y_wall.append(0), position_wall.append('right')  # 25
            x_wall.append(2), y_wall.append(1), position_wall.append('right')  # 26
        elif grid_size == 10:
            x_wall.append(0), y_wall.append(0), position_wall.append('right')  # 101
            x_wall.append(0), y_wall.append(1), position_wall.append('right')  # 102
            x_wall.append(0), y_wall.append(2), position_wall.append('right')  # 103
            x_wall.append(0), y_wall.append(3), position_wall.append('right')  # 104
            x_wall.append(2), y_wall.append(6), position_wall.append('right')  # 105
            x_wall.append(2), y_wall.append(7), position_wall.append('right')  # 106
            x_wall.append(2), y_wall.append(8), position_wall.append('right')  # 107
            x_wall.append(2), y_wall.append(9), position_wall.append('right')  # 108
            x_wall.append(3), y_wall.append(0), position_wall.append('right')  # 109
            x_wall.append(3), y_wall.append(1), position_wall.append('right')  # 110
            x_wall.append(3), y_wall.append(2), position_wall.append('right')  # 111
            x_wall.append(3), y_wall.append(3), position_wall.append('right')  # 112
            x_wall.append(5), y_wall.append(4), position_wall.append('right')  # 113
            x_wall.append(5), y_wall.append(5), position_wall.append('right')  # 114
            x_wall.append(5), y_wall.append(6), position_wall.append('right')  # 115
            x_wall.append(5), y_wall.append(7), position_wall.append('right')  # 116
            x_wall.append(7), y_wall.append(0), position_wall.append('right')  # 117
            x_wall.append(7), y_wall.append(1), position_wall.append('right')  # 118
            x_wall.append(7), y_wall.append(2), position_wall.append('right')  # 119
            x_wall.append(7), y_wall.append(3), position_wall.append('right')  # 120
            x_wall.append(7), y_wall.append(6), position_wall.append('right')  # 121
            x_wall.append(7), y_wall.append(7), position_wall.append('right')  # 122
            x_wall.append(7), y_wall.append(8), position_wall.append('right')  # 123
            x_wall.append(7), y_wall.append(9), position_wall.append('right')  # 124
        else:
            raise NotImplementedError

        x_wall, y_wall, position_wall = np.array(x_wall), np.array(y_wall), np.array(position_wall)
        return x_wall, y_wall, position_wall

    def render(self):
        """
            very slow, should only be used for debugging
        """
        taxi_x, taxi_y = self.objs['taxi']['x'], self.objs['taxi']['y']
        pass_x, pass_y = self.objs['passenger']['x'], self.objs['passenger']['y']
        in_taxi = self.objs['passenger']['in_taxi']
        dest_x, dest_y = self.objs['destination']['x'], self.objs['destination']['y']
        wall_x_l, wall_y_l = self.objs['wall_list']['x'], self.objs['wall_list']['y']

        out_list = []
        out_list.append('+' + (2*self.grid_size - 1)*'-' + '+\n')
        for row in range(self.grid_size):
            row_string = list('|' + (self.grid_size - 1)*' :' + ' |\n')
            for counter, loc in enumerate(self.PREDEFINED_LOCATIONS):
                if row == loc[1]:
                    row_string[loc[0]*2 + 1] = self.POSITION_NAMES[counter]
            if row == taxi_y:
                if not in_taxi:
                    row_string[taxi_x*2 + 1] = colorize(' ', 'yellow', highlight=True)
                else:
                    row_string[taxi_x*2 + 1] = colorize(' ', 'green', highlight=True)
            if row == pass_y and not in_taxi:
                letter = row_string[pass_x*2 + 1]
                row_string[pass_x*2 + 1] = colorize(letter, 'blue', bold=True)
            if row == dest_y:
                letter = row_string[dest_x*2 + 1]
                row_string[dest_x*2 + 1] = colorize(letter, 'magenta')
            if self.grid_size == 5:
                for wall_obj_x, wall_obj_y in zip(wall_x_l[-6:], wall_y_l[-6:]):
                    if row == wall_obj_y:
                        row_string[wall_obj_x*2 + 2] = '|'
            elif self.grid_size == 10:
                for wall_obj_x, wall_obj_y in zip(wall_x_l[-24:], wall_y_l[-24:]):
                    if row == wall_obj_y:
                        row_string[wall_obj_x*2 + 2] = '|'
            else:
                raise NotImplementedError
            out_list.append(''.join(row_string))
        out_list.append('+' + (2*self.grid_size - 1)*'-' + '+\n')
        # reverse out string to have same orientation as in paper
        out_string = ''.join(out_list[::-1])
        print(out_string)
        return

    def human_play(self):
        self.reset()
        state = self.s
        while True:
            self.render()
            print('state: ', state)
            action = input()
            assert 0 <= int(action) <= 5
            print('action: ', TaxiEnvironment.ACTION_MAPPING[int(action)])
            new_state, reward, done, _ = self.step(int(action))
            if done:
                break
            state = new_state
        return


if __name__ == '__main__':
    # oo_env = TaxiEnvironment(grid_size=5, mode='OO MDP')
    # print(len(oo_env.oo_mdp_dict['oo_mdp_to_flat_map']))
    # factored_env = TaxiEnvironment(grid_size=5, mode='factored MDP')
    # print(len(factored_env.factored_mdp_dict['factored_to_flat_map']))
    classical_env = TaxiEnvironment(grid_size=5, mode='classical MDP')
    classical_env.human_play()
