import numpy as np
from utility import *
from copy import deepcopy
from read_data import *
from sklearn.preprocessing import MinMaxScaler


class Agent(object):
    '''Base Class of Agent
    '''

    def __init__(self, initial_env=None, capacity=10000):
        self.env = initial_env
        self.A = []
        self.experience = Experience(capacity=capacity)
        self.S = []

    def get_RC(self):
        actions, reduced_costs = deepcopy(self.env.available_action)
        return reduced_costs
    # get augmented state from the current environment

    def get_aug_state(self):

        actions, reduced_costs = deepcopy(self.env.available_action)
        total_added = len(actions)
        patterns = self.env.current_patterns[:]
        is_action = np.asarray([0]*len(patterns))
        patterns.extend(actions)
        Is_action = np.zeros(len(patterns))
        col_num = len(patterns)
        cons_num = self.env.n
        column_features = []
        cons_features = []
        edge_indices = [[], []]
        # column features, also augment all actions
        Is_action = np.append(Is_action, np.ones(total_added))
        RC = self.env.RC[:]
        RC = np.append(RC, reduced_costs)
        In_Cons_Num = self.env.In_Cons_Num[:]

        actions = np.asarray(actions)
        if actions.ndim == 1:
            In_Cons_Num = np.append(In_Cons_Num, np.count_nonzero(actions))
        else:
            In_Cons_Num = np.append(
                In_Cons_Num, np.count_nonzero(np.asarray(actions), axis=1))

        ColumnSol_Val = self.env.ColumnSol_Val[:]
        ColumnSol_Val = np.append(ColumnSol_Val, np.zeros(total_added))
        Waste = self.env.Waste[:]
        Waste = np.append(Waste, [self.env.roll_len-np.dot(np.asarray(
            actions[i]), np.asarray(self.env.order_lens)) for i in range(total_added)])
        stay_in = self.env.stay_in[:]
        stay_in = np.append(stay_in, np.zeros(total_added))
        stay_out = self.env.stay_out[:]
        stay_out = np.append(stay_out, np.zeros(total_added))
        just_left = self.env.just_left[:]
        just_left = np.append(just_left, np.zeros(total_added))
        just_enter = self.env.just_enter[:]
        just_enter = np.append(just_enter, np.zeros(total_added))
        is_action = np.append(is_action, np.ones(total_added))

        # constraint features, also augment all actions
        Shadow_Price = self.env.Shadow_Price[:]
        In_Cols_Num = self.env.In_Cols_Num[:]
        for action in actions:
            non_zero = np.nonzero(action)
            for idx in non_zero:
                In_Cols_Num[idx] += 1

        RC = np.asarray(RC).reshape(-1, 1)
        Shadow_Price = np.asarray(Shadow_Price).reshape(-1, 1)
        In_Cons_Num = np.asarray(In_Cons_Num).reshape(-1, 1)
        In_Cols_Num = np.asarray(In_Cols_Num).reshape(-1, 1)
        ColumnSol_Val = np.asarray(ColumnSol_Val).reshape(-1, 1)
        Waste = np.asarray(Waste).reshape(-1, 1)
        stay_in = np.asarray(stay_in).reshape(-1, 1)
        stay_out = np.asarray(stay_out).reshape(-1, 1)

        Scaler_RC = MinMaxScaler()
        Scaler_RC.fit(RC)
        RC = Scaler_RC.transform(RC)
        Scaler_SP = MinMaxScaler()
        Scaler_SP.fit(Shadow_Price)
        Shadow_Price = Scaler_RC.transform(Shadow_Price)
        Scaler_IConsN = MinMaxScaler()
        Scaler_IConsN.fit(In_Cons_Num)
        In_Cons_Num = Scaler_IConsN.transform(In_Cons_Num)
        Scaler_IColsN = MinMaxScaler()
        Scaler_IColsN.fit(In_Cols_Num)
        In_Cols_Num = Scaler_IColsN.transform(In_Cols_Num)
        Scaler_CSV = MinMaxScaler()
        Scaler_CSV.fit(ColumnSol_Val)
        ColumnSol_Val = Scaler_CSV.transform(ColumnSol_Val)
        Scaler_W = MinMaxScaler()
        Scaler_W.fit(Waste)
        Waste = Scaler_W.transform(Waste)

        Scaler_si = MinMaxScaler()
        Scaler_si.fit(stay_in)
        stay_in = Scaler_si.transform(stay_in)

        Scaler_out = MinMaxScaler()
        Scaler_out.fit(stay_out)
        stay_out = Scaler_out.transform(stay_out)

        RC = list(RC.T[0])
        Shadow_Price = list(Shadow_Price.T[0])
        In_Cons_Num = list(In_Cons_Num.T[0])
        In_Cols_Num = list(In_Cols_Num.T[0])
        ColumnSol_Val = list(ColumnSol_Val.T[0])
        Waste = list(Waste.T[0])
        stay_in = list(stay_in.T[0])
        stay_out = list(stay_out.T[0])

        # constraint nodes
        for j in range(cons_num):
            con_feat = []
            con_feat.append(Shadow_Price[j])
            con_feat.append(In_Cols_Num[j])
            cons_features.append(con_feat)

        # normalize here for each information
        for i in range(col_num):
            col_feat = []
            col_feat.append(RC[i])
            col_feat.append(In_Cons_Num[i])
            col_feat.append(ColumnSol_Val[i])
            col_feat.append(Waste[i])
            col_feat.append(stay_in[i])
            col_feat.append(stay_out[i])
            col_feat.append(just_left[i])
            col_feat.append(just_enter[i])
            col_feat.append(is_action[i])

            column_features.append(col_feat)

        for m in range(len(patterns)):
            for n in range(len(patterns[0])):
                if patterns[m][n] != 0:
                    # then mth column is connected to nth cons
                    edge_indices[0].append(m)
                    edge_indices[1].append(n)

        edge_indices = np.asarray(edge_indices)
        edge_indices[[0, 1]] = edge_indices[[1, 0]]
        cons_features = np.asarray(cons_features)
        column_features = np.asarray(column_features)

        # need this total_added for reading the Q values, need actions to select onne pattern after read Q values
        aug_state, action_info = (
            (cons_features, edge_indices, column_features), (total_added, actions))
        return aug_state, action_info

    def act(self, a0):
        # get the current super state
        s0_augmented, action_info_0 = self.S
        total_0 = deepcopy(action_info_0[0])
        # step change the environnment, update all the information used for agent to construct state
        r, is_done = self.env.step(a0, True)
        s1_augmented, action_info_1 = self.get_aug_state()
        total_1 = action_info_1[0]
        trans = Transition(s0_augmented, a0, r[-1], is_done, s1_augmented,
                           action_info_0, total_0, total_1, self.env.useful_pattern, r[:-1])
        total_reward = self.experience.push(trans)
        self.S = s1_augmented, action_info_1
        return s1_augmented, r[-1], is_done, total_reward

    def _decayed_epsilon(self, cur_episode: int,
                         min_epsilon: float,
                         max_epsilon: float,
                         target_episode: int) -> float:
        slope = (min_epsilon - max_epsilon) / (target_episode)
        intercept = max_epsilon
        return max(min_epsilon, slope * cur_episode + intercept)

    def sample(self, batch_size=32):

        return self.experience.sample(batch_size)

    @property
    def total_trans(self):

        return self.experience.total_trans

    def last_episode_detail(self):
        self.experience.last_episode.print_detail()
