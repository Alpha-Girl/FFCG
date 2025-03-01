import numpy as np
from env import *
from DQN import *
from read_data import *
import time
import os
import Parameters


def follow_policy(DQN, action_info, s):
    '''multiple actions can be returned
    '''
    total_added, Actions = action_info
    Q_s = DQN.target_Q(s)
    Q_s_for_action = Q_s[-total_added::]
    return_list = []
    Q_s_for_action = np.round(Q_s[-total_added::])
    for i in range(len(Q_s_for_action)):
        if Q_s_for_action[i] > 0.5:
            return_list.append(Actions[i])
    if len(return_list) == 0:
        Q_s_for_action = Q_s[-total_added::]
        idx = int(np.argmax(Q_s_for_action))
        return_list.append(Actions[idx])
    return return_list


def test_ffcg(FFCG_agent, model_index):

    result_path = Parameters.RESULT_PATH
    test_data_path = Parameters.DATA_PATH
    names = os.listdir(test_data_path)
    total_length = len(names)
    FFCG_agent.target_Q.restore_state(
        Parameters.MODEL_PATH+'Model_' + str(model_index)+'.pt')
    FFCG_agent.behavior_Q.restore_state(
        Parameters.MODEL_PATH+'Model_' + str(model_index)+'.pt')
    for i in range(total_length):
        FFCG_p_list = []
        time_start = time.time()
        cut2 = instance_test(i, names[i])
        _, is_done = cut2.initialize(1)
        FFCG_p_list.append(len(cut2.current_patterns))
        FFCG_agent.env = cut2
        FFCG_agent.S = FFCG_agent.get_aug_state()
        while True:
            if is_done:
                break
            action_info = FFCG_agent.S[1]
            s = FFCG_agent.S[0]
            action = follow_policy(FFCG_agent, action_info, s)
            _, is_done = cut2.step_new(action, False)
            FFCG_agent.S = FFCG_agent.get_aug_state()
            FFCG_p_list.append(len(cut2.current_patterns))
        history_opt_FFCG = cut2.objVal_history
        time_end = time.time()
        obj_FFCG = history_opt_FFCG[-1]
        steps_FFCG = len(history_opt_FFCG)
        print("FFCG takes {} steps to reach obj {:.2f}  with time {:.2f}".format(
            steps_FFCG, obj_FFCG, time_end-time_start))
