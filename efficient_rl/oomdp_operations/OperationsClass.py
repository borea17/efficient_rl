import numpy as np


class Operations:

    @staticmethod
    def add_hypotheses(h1, h2):
        """
            elementwise addition of two conditions h1 and h2 using the commutative operator
            defined by Diuk
        """
        if h1 is None:
            return h2
        h3 = []
        for elem1, elem2 in zip(h1, h2):
            if elem1 == elem2:  # 0+0 = 0 and 1+1 = 1
                h3.append(elem1)
            else:  # 0/1 + * = * and 0 + 1 = *
                h3.append(np.nan)
        return np.array(h3)

    @staticmethod
    def hypothesis_matches(h1, h2):
        """
            returns True when condition h1 matches condition h2, otherwise False
            NOTE: this operation is not commutative
        """
        if h1 is None:
            return False
        return ((h1 == h2).sum() + np.isnan(h1).sum()) == len(h1)

    @staticmethod
    def apply_effect_to_attribute(E, att):
        """
            applies an arbitary effect of E to attribute att
            NOTE: all effects yield the same result (see check_for_incombatible_effects)
        """
        effect_type, effect = E.popitem()
        if effect_type == 'addition':
            prediction = att + effect
        elif effect_type == 'assignment':
            prediction = effect
        elif effect_type == 'multiplication':
            prediction = att * effect
        elif effect_type == 'no effect':  # only DOORmax can assign this effect type
            prediction = att
        return prediction

    @staticmethod
    def eff(state, new_state, att, effect_types):
        """
            returns a dictionary of all effect types and their corresponding effect that would
            transform attribute att in state to its value in new_state
        """
        possible_effects = []
        for effect_type in effect_types:
            if effect_type == 'addition':
                effect = new_state[att] - state[att]
            elif effect_type == 'assignment':
                effect = new_state[att]
            elif effect_type == 'multiplication':
                if state[att] == 0 or new_state[att] == 0:  # case invalid or covered by assignment
                    effect = None
                else:
                    effect = new_state[att] / state[att]
            possible_effects.append(effect)
        return possible_effects

    @staticmethod
    def argwhere_conds_match_h1(list_of_conds, h1):
        matching = []
        for index, h2 in enumerate(list_of_conds):
            if Operations.hypothesis_matches(h2, h1):
                matching.append(index)
        return np.array(matching)

    @staticmethod
    def argwhere_h1_matches_conds(h1, conds):
        matching = []
        for index, h2 in enumerate(conds):
            if Operations.hypothesis_matches(h1, h2):
                matching.append(index)
        return np.array(matching)
