import gym
import networkx as nx
import matplotlib.pyplot as plt


class FactoredTaxi:
    """
    taxi environment as a factored MDP, see p. 37 of Diuks Dissertation
    """

    ACTION_MAPPING = {0: 'South', 1: 'North', 2: 'East', 3: 'West', 4: 'Pickup', 5: 'Dropoff'}

    def __init__(self):
        self.env = gym.make('Taxi-v3').env
        self.DBNs = FactoredTaxi.create_DBNs()
        return

    def step(self, action):
        if action == 5:  # drop off action
            taxi_row, taxi_col, pass_loc, dest_loc = list(self.env.decode(self.env.s))
            if (taxi_row, taxi_col) in self.env.locs:  # taxi location on any predefined locations
                if self.env.locs.index((taxi_row, taxi_col)) != dest_loc:
                    # illegal drop off action following Diuk
                    new_state = self.convert_state_into_factored_state()
                    reward = -10
                    done = False
                    info = None
                    self.env.lastaction = 5
                    return new_state, reward, done, info
            new_state, reward, done, info = self.env.step(action)
        else:
            new_state, reward, done, info = self.env.step(action)
        new_state = self.convert_state_into_factored_state()
        self.s = new_state
        return new_state, reward, done, info

    def reset(self):
        self.env.reset()
        self.s = self.convert_state_into_factored_state()
        return self.s

    def render(self):
        self.env.render()
        return

    def encode(self, taxi_row, taxi_col, pass_loc, dest_idx):
        self.env.s = self.env.encode(taxi_row, taxi_col, pass_loc, dest_idx)
        self.s = self.convert_state_into_factored_state()
        return self.s

    def convert_state_into_factored_state(self):
        """
        converts the state (int) into a 4-tuple:
        <y_loc_taxi, x_loc_taxi, pass_loc, pass_dest>
        """
        factored_state_tuple = [elem for elem in self.env.decode(self.env.s)]
        return factored_state_tuple

    def make_flat_state(self):
        return self.env.s

    @staticmethod
    def create_DBNs():
        DBNs = dict()
        DBNs['reward'] = FactoredTaxi.create_reward_DBN_for_each_action()
        DBNs['transition'] = FactoredTaxi.create_transition_DBN_for_each_action()
        return DBNs

    @staticmethod
    def create_transition_DBN_for_each_action():
        """
        dependencies can be used to create DBN,
        storing order corresponds to action mapping and state of openai
        """
        dependencies_south = [[0], [1], [2], [3]]
        dependencies_north = [[0], [1], [2], [3]]
        dependencies_east = [[0], [0, 1], [2], [3]]
        dependencies_west = [[0], [0, 1], [2], [3]]
        dependencies_pickup = [[0], [1], [0, 1, 2], [3]]
        # dependencies_dropoff = [[0], [1], [0, 1, 2], [3]]  # gym drop off dependency
        dependencies_dropoff = [[0], [1], [0, 1, 2, 3], [3]]  # Diuk drop off dependency
        dependencies = [dependencies_south, dependencies_north, dependencies_east,
                        dependencies_west, dependencies_pickup, dependencies_dropoff]
        transition_DBNs = []
        state_t = ['y_loc', 'x_loc', 'pass_loc', 'dest_loc']
        state_tp1 = [state for state in range(4)]
        for dependency in dependencies:
            G = nx.DiGraph()
            G.add_nodes_from(state_t, bipartite=0)
            G.add_nodes_from(state_tp1, bipartite=1)
            for i_end in range(len(state_tp1)):
                G.nodes[i_end]['dependency'] = dependency[i_end]
                for i_start in dependency[i_end]:
                    G.add_edges_from([(state_t[i_start], state_tp1[i_end])])
            transition_DBNs.append(G)
        return transition_DBNs

    @staticmethod
    def create_reward_DBN_for_each_action():
        dependencies_south = [0]
        dependencies_north = [0]
        dependencies_east = [0]
        dependencies_west = [0]
        dependencies_pickup = [0, 1, 2, 3]
        dependencies_dropoff = [0, 1, 2, 3]
        dependencies = [dependencies_south, dependencies_north, dependencies_east,
                        dependencies_west, dependencies_pickup, dependencies_dropoff]

        reward_DBNs = []
        state_t = ['y_loc', 'x_loc', 'pass_loc', 'dest_loc']
        expected_reward = ['reward']
        for dependency in dependencies:
            G = nx.DiGraph()
            G.add_nodes_from(state_t, bipartite=0)
            G.add_nodes_from(expected_reward, bipartite=1)
            G.nodes['reward']['dependency'] = dependency
            for i_start in dependency:
                G.add_edges_from([(state_t[i_start], expected_reward[0])])
            reward_DBNs.append(G)
        return reward_DBNs

    def draw_DBN(self, action):
        action_string = FactoredTaxi.ACTION_MAPPING[action]
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=[10, 5])
        fig.suptitle('Action ' + action_string)
        # transition
        plt.subplot(ax1)
        G = self.DBNs['transition'][action]
        top_nodes = [n for n, d in G.nodes(data=True) if d['bipartite'] == 0]
        bottom_nodes = [n for n, d in G.nodes(data=True) if d['bipartite'] == 1]
        pos = dict()
        pos.update((n, (i, 2)) for i, n in enumerate(top_nodes))
        pos.update((n, (i, 1)) for i, n in enumerate(bottom_nodes))
        nx.draw_networkx(G, pos=pos, node_size=1450, font_size=10)
        plt.title('Transition Dynamics')
        # reward
        plt.subplot(ax2)
        G_r = self.DBNs['reward'][action]
        top_nodes = [n for n, d in G_r.nodes(data=True) if d['bipartite'] == 0]
        bottom_nodes = [n for n, d in G_r.nodes(data=True) if d['bipartite'] == 1]
        pos = dict()
        pos.update((n, (i, 2)) for i, n in enumerate(top_nodes))
        pos.update((n, (i, 1)) for i, n in enumerate(bottom_nodes))
        nx.draw_networkx(G_r, pos=pos, node_size=1450, font_size=10)
        plt.title('Reward Dynamics')
        plt.show()
        return

    def human_play(self):
        self.env.reset()
        while True:
            self.env.render()
            action = input()
            new_state, reward, done, _ = env.step(int(action))
            print(new_state)
            print('y_loc:', new_state[0], ', x_loc:', new_state[1], ', pass_loc', new_state[2],
                  'dest_loc', new_state[3])
            if done:
                break
        return


if __name__ == '__main__':
    env = FactoredTaxi()
    env.human_play()
    # num_actions = 6
    # for i_action in range(num_actions):
    #     env.draw_DBN(i_action)
    
