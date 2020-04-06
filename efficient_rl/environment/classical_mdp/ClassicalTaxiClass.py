from gym.envs.toy_text import taxi


class ClassicalTaxi(taxi.TaxiEnv):

    """
        in Diuks version of Taxi, a passenger can only be dropped off if the taxi is at
        destination, in the typical gym environment a passenger can be dropped off at any of the
        predefined destinations

        therefore this class overloads the standard step
    """
    
    def __init__(self):
        super().__init__()
        return

    def step(self, action):
        if action == 5:  # drop off action
            taxi_row, taxi_col, pass_loc, dest_loc = list(self.decode(self.s))
            if (taxi_row, taxi_col) in self.locs:  # taxi location on any predefined destination
                if self.locs.index((taxi_row, taxi_col)) != dest_loc:
                    # illegal dropoff action following Diuk
                    new_state = self.s
                    reward = - 10
                    done = False
                    info = None
                    self.lastaction = 5
                    return new_state, reward, done, info
            # taxi location not on any predefined destination or at target destination
            new_state, reward, done, info = super().step(action)
        else:
            new_state, reward, done, info = super().step(action)
        return new_state, reward, done, info

    def human_play(self):
        self.reset()
        while True:
            self.render()
            action = input()
            new_state, reward, done, _ = self.step(int(action))
            if done:
                break
        return


if __name__ == '__main__':
    env = ClassicalTaxi()
    env.human_play()
