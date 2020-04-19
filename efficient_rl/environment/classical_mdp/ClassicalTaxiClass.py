from gym.envs.toy_text import taxi
from gym.utils import seeding


class ClassicalTaxi(taxi.TaxiEnv):

    """
        in Diuks version of Taxi, a passenger can only be dropped off if the taxi is at
        destination, in the typical gym environment a passenger can be dropped off at any of the
        predefined destinations
        therefore this class overloads the standard step
    """

    def __init__(self, standard_reset=True):
        super().__init__()
        if standard_reset:  # see reset function
            self.replace = True
        else:
            self.replace = False
        return

    def step(self, action):
        if action == 5:  # drop off action
            taxi_row, taxi_col, pass_loc, dest_loc = list(self.decode(self.s))
            if (taxi_row, taxi_col) in self.locs:  # taxi location on any predefined destination
                if self.locs.index((taxi_row, taxi_col)) != dest_loc:
                    # illegal dropoff action following Diuk
                    new_state = self.s
                    reward = -10
                    done = False
                    info = None
                    self.lastaction = 5
                    return new_state, reward, done, info
            # taxi location not on any predefined destination or at target destination
            new_state, reward, done, info = super().step(action)
        else:
            new_state, reward, done, info = super().step(action)
        return new_state, reward, done, info

    def reset(self):
        """
            in gym reset, passenger location and destination location are never the same,
            in original of Dietterich this is possible
        """
        rng, seed = seeding.np_random()

        taxi_locs = [0, 1, 2, 3, 4]
        pass_dest_locs = [0, 1, 2, 3]

        taxi_row, taxi_column = rng.choice(taxi_locs, size=2, replace=True)
        pass_loc, dest_loc = rng.choice(pass_dest_locs, size=2, replace=self.replace)
        self.s = self.encode(taxi_row, taxi_column, pass_loc, dest_loc)
        return self.s

    def human_play(self):
        self.reset()
        while True:
            self.render()
            action = input()
            new_state, reward, done, _ = self.step(int(action))
            print(reward)
            if done:
                break
        return


if __name__ == '__main__':
    env = ClassicalTaxi()
    env.human_play()
