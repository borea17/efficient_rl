from efficient_rl.oomdp_operations import Operations as oo_mdp_OP


class CRLearner:

    ACTION_MAPPING = {0: 'South', 1: 'North', 2: 'East', 3: 'West', 4: 'Pickup', 5: 'Dropoff'}

    def __init__(self, index_action, env_name='Taxi'):
        if env_name == 'Taxi':
            self.action = CRLearner.ACTION_MAPPING[index_action]
        else:
            self.action = index_action

        self.models, self.rewards = [], []
        return

    def add_experience(self, condition, reward):
        if reward in self.rewards:  # rewards has already been achieved for this action
            i_reward = self.rewards.index(reward)
            self.models[i_reward] = oo_mdp_OP.add_hypotheses(self.models[i_reward], condition)
        else:
            self.models.append(condition)
            self.rewards.append(reward)
        return

    def can_predict(self, condition):
        i_match_condition = oo_mdp_OP.argwhere_conds_match_h1(self.models, condition)
        if len(i_match_condition) != 1:
            return False
        else:
            self.i_match_condition = i_match_condition[0]
            return True

    def predict(self, condition):
        return self.rewards[self.i_match_condition]
