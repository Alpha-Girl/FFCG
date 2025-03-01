import Parameters
import numpy as np
from agents import Agent
from net import BipartiteGNN
from copy import deepcopy


class DQNAgent(Agent):
    '''
    '''

    def __init__(self, env,
                 capacity,
                 batch_size,
                 epochs,
                 embedding_size,
                 cons_num_features,
                 vars_num_features,
                 learning_rate):

        super(DQNAgent, self).__init__(env, capacity)
        self.embedding_size = embedding_size
        self.cons_num_features = cons_num_features
        self.vars_num_features = vars_num_features
        self.lr = learning_rate
        self.behavior_Q = BipartiteGNN(embedding_size=self.embedding_size, cons_num_features=self.cons_num_features,
                                       vars_num_features=self.vars_num_features, learning_rate=self.lr)
        self.target_Q = BipartiteGNN(embedding_size=self.embedding_size, cons_num_features=self.cons_num_features,
                                     vars_num_features=self.vars_num_features, learning_rate=self.lr)
        self._update_target_Q()
        self.batch_size = batch_size
        self.epochs = epochs

    def _update_target_Q(self):

        self.target_Q.set_weights(deepcopy(self.behavior_Q.get_weights()))

    # s is the super s0, A is the list containing all actions

    def policy(self, action_info, s, epsilon=None):

        total_added, Actions = action_info
        Q_s = self.behavior_Q(s)
        Q_s_for_action = Q_s[-total_added::]
        rand_value = np.random.random()
        if epsilon is not None and rand_value < epsilon:
            myarray = np.random.randint(0, 2, len(Actions))
            while sum(myarray) == 0:
                myarray = np.random.randint(0, 2, len(Actions))
            return_list = []
            for i in range(len(myarray)):
                if myarray[i] == 1:
                    return_list.append(Actions[i])
            return return_list
        else:
            return_list = []
            Q_s_for_action = np.round(Q_s[-total_added::])
            for i in range(len(Q_s_for_action)):
                if Q_s_for_action[i] > Parameters.threshold:
                    return_list.append(Actions[i])
            if len(return_list) == 0:
                Q_s_for_action = Q_s[-total_added::]
                idx = int(np.argmax(Q_s_for_action))
                return_list.append(Actions[idx])
            return return_list

    # s is the super s0, A is the list containing all actions
    # need action info 0 and total 1 (total_1 to get max Q_1, action_info_0 to get update index)

    def get_max(self,  total_1, s):
        Q_s = self.target_Q.call(s)
        Q_s_for_action = Q_s[-total_1::]
        return np.max(Q_s_for_action)

    # this method is used to get target in _learn_from_memory function
    def _learn_from_memory(self):
        # trans_pieces is a list of transitions
        trans_pieces = self.sample(self.batch_size)  # Get transition data
        # as s0 is a list, so vstack
        states_0 = np.vstack([x.s0 for x in trans_pieces])
        actions_0 = np.array([x.a0 for x in trans_pieces])
        reward_1 = np.array([x.reward for x in trans_pieces])
        action_info = np.vstack([x.action_info_0 for x in trans_pieces])
        totals_0 = np.vstack([x.total_0 for x in trans_pieces])
        useful_patterns = np.array([x.useful_patterns for x in trans_pieces])
        val_list = np.array([x.val_list for x in trans_pieces])
        y_batch = []
        for i in range(len(states_0)):
            # get the index of action that is taken at s0
            acts_0 = action_info[i][1]
            idx = 0
            idx_list = []
            jdx_list = []
            jdx = 0
            for act in acts_0:
                for j in range(len(actions_0[i])):
                    act_0 = list(actions_0[i][j])

                    if (act == act_0).all():
                        idx_list.append(idx)
                        jdx_list.append(jdx)
                        jdx += 1
                idx += 1
            y = self.target_Q.call(states_0[i]).numpy()
            # set the non action terms to be 0
            y[0:-totals_0[i][0]] = 0
            Q_target = reward_1[i]+Parameters.beta * \
                (len(acts_0)-np.sum(useful_patterns[i][-len(acts_0):]))
            for k in range(len(actions_0[i])):
                j = idx_list[k]
                if useful_patterns[i][-totals_0[i][0]+j] == 1:
                    y[-totals_0[i][0]+j] = max(Q_target, 0) * \
                        val_list[i][k]/(np.sum(val_list[i])+0.01)
                else:
                    y[-totals_0[i][0]+j] = -Parameters.beta
            y_batch.append(np.asarray(y))
        y_batch = np.asarray(y_batch)
        X_batch = states_0
        loss = self.behavior_Q.train_or_test(
            X_batch, y_batch, totals_0, actions_0, action_info, True)
        self._update_target_Q()
        return loss
