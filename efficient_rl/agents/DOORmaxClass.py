from efficient_rl.agents import RmaxBaseAgent
from efficient_rl.agents.oo_mdp_learner import CELearner, CRLearner
from efficient_rl.oo_mdp_operations import Operations as oo_mdp_OP
import itertools
import numpy as np


class DOORmax(RmaxBaseAgent):
    """
    Deterministic Object-Oriented Rmax agent, designed for  deterministic Propositional OO-MDPs:
        - for each action and a given condition there occurs only one effect with probability 1
    """

    allowed_effect_types = ['addition', 'assignment', 'multiplication']

    def __init__(self, nS, nA, r_max, gamma, env_name, k, eff_types, num_atts, delta, oo_mdp_dict):
        M = 1  # deterministic
        super().__init__(nS, nA, gamma, M, r_max, env_name, delta)
        assert all([eff in DOORmax.allowed_effect_types for eff in eff_types])
        self.num_atts = num_atts
        self.k = k  # maximum number of different effects possible for any action, att, effect type
        self.effect_types = eff_types
        # fictious state from which maximum reward can be obtained
        self.s_max = -1
        self.effect_type_mapping = {eff_type: index for index, eff_type in enumerate(eff_types)}
        # initialize transition and reward learner
        self.transition_F_a, self.transition_F_att_a = self.initialize_transition_failure_conds()
        self.CELearners = self.initialize_CELearners()
        self.reward_F_a, self.no_effect_rewards = self.initialize_reward_failure_conds()
        self.CRLearners = self.initialize_CRLearners()
        # map oo_mdp state to flat_state and vice versa through dictionary
        self.oo_mdp_dict = oo_mdp_dict
        # enhance update procedure
        self.all_possible_state_action_seq = set((state, action) for state, action in
                                                 itertools.product(self.states, self.actions))
        self.updated_state_action_seq = set()
        return

    def step(self, state_cond, deterministic=False):
        flat_state = self.make_flat_state(state_cond)
        # Compute a near optimal value function (blowing up here)
        q_optimal = self.compute_near_optimal_value_function()
        # Observe current state and pick action greedily
        if deterministic:
            # pick first action that is optimal
            action = np.argmax(q_optimal[flat_state])
        else:
            # allow for random decision if multiple actions are optimal
            best_actions = np.argwhere(q_optimal[flat_state] == np.amax(q_optimal[flat_state]))
            action = best_actions[np.random.randint(len(best_actions))][0]
        return action

    def update_emp_MDP(self, state_cond_not_used, action_not_used):
        states_actions_to_check = self.all_possible_state_action_seq - self.updated_state_action_seq
        for flat_state, action in states_actions_to_check:
            state_conds = self.oo_mdp_dict['flat_to_oo_mdp_map'][flat_state]
            for state_cond in state_conds:
                if self.reward_learner_can_predict(state_cond, action) and \
                   self.transition_learner_can_predict(state_cond, action):
                    # update model only if both transition adn reward learner can predict
                    self.emp_reward_dist[flat_state, action] = \
                        self.predict_expected_immediate_reward(state_cond, action)
                    self.emp_transition_dist[action, flat_state] = \
                        self.predict_transition_probs(state_cond, action)
                    self.updated_state_action_seq.add((flat_state, action))
                    break
        return

    def add_experience_to_transition_learner(self, state_cond, action, new_state_cond):
        # extract state and condition from state_cond
        state, cond, cond_list = state_cond[0], state_cond[1], list(state_cond[1])
        # extract new_state from new_state_cond
        new_state = new_state_cond[0]
        if np.all(state == new_state):  # action had no effect: found failure condition
            # NOTE: cannot use the `add_hypotheses` operation here
            self.transition_F_a[action].append(cond_list)
        else:  # action had an effect (but probably not on all attributes changed)
            for i_att, (att, new_att) in enumerate(zip(state, new_state)):
                if att == new_att:  # attribute did not change
                    # NOTE: cannot use the `add_hypotheses` operation here (negative example)
                    if cond_list not in self.transition_F_att_a[action, i_att]:
                        self.transition_F_att_a[action, i_att].append(cond_list)
                    continue
                # find all possible effects that explain the attribute change
                possible_effects = oo_mdp_OP.eff(state, new_state, i_att, self.effect_types)
                for i_eff_type, eff in enumerate(possible_effects):
                    # retrieve effect and corresponding index of effect type
                    # NOTE: Not all effects in self.effect_type occur necessarily as poss. effects
                    if eff is None:  # for this effect type no effect could be calculated
                        self.CELearners[action][i_att][i_eff_type].remove()
                        continue
                    # update CELearner
                    self.CELearners[action][i_att][i_eff_type].add_experience(cond, eff)

                    # ASSUMPTION 1: For each action and attribute, only effects of one type occur
                    # ASSUMPTION 2: At most k different pairs in each CELearner
                    # REASONING: otherwise we could learn a different effect for each condition
                    if len(self.CELearners[action][i_att][i_eff_type]) > self.k:
                        self.CELearners[action][i_att][i_eff_type].remove()
                    if self.CELearners[action][i_att][i_eff_type].conditions_overlap():
                        self.CELearners[action][i_att][i_eff_type].remove()
        return

    def predict_transition(self, state_cond, action):
        return self.transition_prediction

    def transition_learner_can_predict(self, state_cond, action):
        """
            in addition to returning whether the transition learner can predict the next state,
            this function stores the prediction of next state
        """
        state, cond, cond_list = state_cond[0], state_cond[1], list(state_cond[1])
        self.transition_prediction = self.s_max
        if cond_list in self.transition_F_a[action]:
            # current condition is a known failure condition (no attribute change at all)
            new_state = state
        else:
            new_state = []
            for i_att, att in enumerate(state):
                if cond_list in self.transition_F_att_a[action, i_att]:
                    # condition is a known action-attribute failure condition
                    new_state.append(att)
                    continue

                # at least one of the CELearners for i_att, action needs to be able to predict
                can_predict = [self.CELearners[action, i_att, i_eff].can_predict(cond, state)
                               for i_eff in range(len(self.effect_types))]
                if sum(can_predict) < 1:
                    return False
                else:
                    new_att_preds = [self.CELearners[action, i_att, i_eff].predict(cond, state)
                                     for i_eff, can_pred in enumerate(can_predict) if can_pred == 1]
                    # check whether predictions are consistent
                    if new_att_preds.count(new_att_preds[0]) == len(new_att_preds):
                        new_state.append(new_att_preds[0])
                    else:  # found inconsistent predictions
                        return False
        self.transition_prediction = new_state
        return True

    def predict_transition_probs(self, state_cond, action):
        new_oo_mdp_state = self.predict_transition(state_cond, action)
        flat_state = self.make_flat_state((new_oo_mdp_state, 0))
        transition_probs = np.zeros([len(self.states)])
        transition_probs[flat_state] = 1
        return transition_probs

    def reward_learner_can_predict(self, state_cond, action):
        # extract condition from state_cond
        cond, cond_list = state_cond[1], list(state_cond[1])
        return cond_list in self.reward_F_a[action] or self.CRLearners[action].can_predict(cond)

    def add_experience_to_reward_learner(self, state_cond, action, reward):
        # extract condition from state_cond
        cond, cond_list = state_cond[1], list(state_cond[1])
        if cond_list in self.transition_F_a[action]:
            # no effect actions are treated as negative examples -> add_hypotheses not allowed
            self.reward_F_a[action].append(cond_list)
            self.no_effect_rewards[action] = reward
        else:
            self.CRLearners[action].add_experience(cond, reward)
        return

    def predict_expected_immediate_reward(self, state_cond, action):
        # extract condition from state_cond
        cond, cond_list = state_cond[1], list(state_cond[1])
        if cond_list in self.reward_F_a[action]:
            return self.no_effect_rewards[action]
        else:
            return self.CRLearners[action].predict(cond)

    def initialize_CELearners(self):
        num_actions, num_effs = len(self.actions), len(self.effect_types)
        CELearners = np.empty([num_actions, self.num_atts, num_effs], dtype=object)
        for i_action in self.actions:
            for i_att in range(self.num_atts):
                for i_eff, eff_type in enumerate(self.effect_types):
                    CELearners[i_action, i_att, i_eff] = CELearner(eff_type, i_att, i_action)
        return CELearners

    def initialize_transition_failure_conds(self):
        transition_F_a = np.empty([len(self.actions)], dtype=object)
        transition_F_att_a = np.empty([len(self.actions), self.num_atts], dtype=object)
        for i_action in range(len(self.actions)):
            transition_F_a[i_action] = []
            for i_att in range(self.num_atts):
                transition_F_att_a[i_action, i_att] = []
        return transition_F_a, transition_F_att_a

    def initialize_CRLearners(self):
        CRLearners = np.empty([len(self.actions)], dtype=object)
        for i_action in self.actions:
            CRLearners[i_action] = CRLearner(i_action)
        return CRLearners

    def initialize_reward_failure_conds(self):
        reward_F_a = np.empty([len(self.actions)], dtype=object)
        no_effect_rewards = np.zeros([len(self.actions)])
        for i_action in self.actions:
            reward_F_a[i_action] = []
        return reward_F_a, no_effect_rewards

    def make_flat_state(self, state_cond):
        oo_mdp_state = state_cond[0]
        flat_state = self.oo_mdp_dict['oo_mdp_to_flat_map'][tuple(oo_mdp_state)]
        return flat_state

    def reset(self):
        super().reset()
        # initialize transition and reward learner
        self.transition_F_a, self.transition_F_att_a = self.initialize_transition_failure_conds()
        self.CELearners = self.initialize_CELearners()
        self.reward_F_a, self.no_effect_rewards = self.initialize_reward_failure_conds()
        self.CRLearners = self.initialize_CRLearners()
        # to enhance update procedure
        self.all_possible_state_action_seq = set((state, action) for state, action in
                                                 itertools.product(self.states, self.actions))
        self.updated_state_action_seq = set()
        return
