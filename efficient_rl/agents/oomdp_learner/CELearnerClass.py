import itertools
from efficient_rl.oomdp_operations import Operations as oo_mdp_OP


class CELearner:

    ACTION_MAPPING = {0: 'South', 1: 'North', 2: 'East', 3: 'West', 4: 'Pickup', 5: 'Dropoff'}

    def __init__(self, effect_type, index_attribute, index_action, env_name='Taxi'):
        self.effect_type = effect_type
        self.i_att = index_attribute
        if env_name == 'Taxi':
            self.action = CELearner.ACTION_MAPPING[index_action]
        else:
            self.action = index_action
        self.removed = False

        self.models, self.effects = [], []
        return

    def add_experience(self, condition, effect):
        """
            updates the CELearner given that an effect occured (positive example)
        """
        if self.removed:  # agent already concluded that this CELearner can be removed
            return

        if effect not in self.effects:  # new effect observed
            self.models.append(condition)
            self.effects.append(effect)
        elif effect in self.effects:  # eff already known
            eff_match_i = self.effects.index(effect)
            # update hypothesis of effect
            self.models[eff_match_i] = oo_mdp_OP.add_hypotheses(self.models[eff_match_i], condition)
        return

    def conditions_overlap(self):
        for c1, c2 in itertools.combinations(self.models, 2):
            if oo_mdp_OP.hypothesis_matches(c1, c2) and oo_mdp_OP.hypothesis_matches(c2, c1):
                return True
        return False

    def can_predict(self, condition, state):
        """
            checks whether a consistent prediction can be made, i.e., there is exactly one matching
            condition in self.models
        """
        if self.removed:  # CELearner has been removed
            return False
        else:
            # index indicates which models match conditions
            i_match_condition = oo_mdp_OP.argwhere_conds_match_h1(self.models, condition)
            if len(i_match_condition) != 1:
                return False
            else:
                self.i_match_condition = i_match_condition[0]
                return True

    def predict(self, condition, state):
        """
            returns a prediction for the new attribute in new state,
            NOTE: the matching index is taken from `can_predict(condition, state)` as this function
                  shall only be called when `can_predict(condition, state)` returns True
        """
        if self.effect_type == 'addition':
            prediction = state[self.i_att] + self.effects[self.i_match_condition]
        elif self.effect_type == 'assignment':
            prediction = self.effects[self.i_match_condition]
        elif self.effect_type == 'multiplication':
            prediction = state[self.i_att] * self.effects[self.i_match_condition]
        return prediction

    def __len__(self):
        return len(self.models)

    def remove(self):
        self.models, self.effects = [], []
        self.removed = True
        return
