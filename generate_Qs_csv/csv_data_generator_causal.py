import datetime
import tensorflow as tf  # pip install tensorflow==1.15, the 2.0 version by default does not work
import os
import scipy.io as sio
import numpy as np
import copy
import random

import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + "/..")

from nn_structure.cvrnn import CVRNN
from nn_structure.lstm_Qs_nn import TD_Prediction
from nn_structure.lstm_score_diff_nn import Diff_Prediction

from config.LSTM_Qs_config import LSTMQsCongfig
from config.cvrnn_config import CVRNNCongfig

from utils import handle_trace_length, get_together_training_batch, compromise_state_trace_length
from configuration import MODEL_TYPE, MAX_TRACE_LENGTH, FEATURE_NUMBER, BATCH_SIZE, GAMMA, H_SIZE, \
    model_train_continue, FEATURE_TYPE, ITERATE_NUM, learning_rate, SPORT, action_all

trained_model_name = 'Ice-Hockey-game--10501'

# save_mother_dir = "/cs/oschulte/xiangyus/DRL-ice-hockey-saves"
save_mother_dir = "/Users/shawnxys/Desktop"

# DATA_STORE = "/cs/oschulte/xiangyus/2019-icehockey-data-preprocessed/2018-2019"
DATA_STORE = "/Users/shawnxys/Development/Data/2019-icehockey-data-preprocessed/2018-2019"

SAVED_NETWORK = './models/'

directory_generated_Q_data = "Q_data"

Q_data_DIR = save_mother_dir + '/' + directory_generated_Q_data

DIR_GAMES_ALL = os.listdir(DATA_STORE)

timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
Q_file_name = 'Q_' + str(timestamp) + '.csv'


def get_action_index_in_feature(action_index):
    # the first 12 elements are state features
    return 12 + action_index


# convert s2, s1, s0(current), padding, padding, ...
# to padding, padding, ..., s2, s1, s0(current)
def convert_to_padding_front(padding_end_state):
    padding_front_state = []
    for event in padding_end_state:
        # if both event home and away are 0, it's paddings
        home = event[9]
        away = event[10]

        # the event is padding if both home and away are 0s, since real home and away cannot be the same value and also normalized.
        if home == 0 and away == 0:
            padding_front_state.insert(0, event)
        else:
            padding_front_state.append(event)

    return padding_front_state


# convert padding, padding, ..., s2, s1, s0(current)
# back to s2, s1, s0(current), padding, padding, ...
def convert_back_to_padding_end(padding_front_state):
    padding_end_state = []
    paddings = []
    for event in padding_front_state:
        # if both event home and away are 0, it's paddings
        home = event[9]
        away = event[10]

        # the event is padding if both home and away are 0s, since real home and away cannot be the same value and also normalized.
        if home == 0 and away == 0:
            # padding_front_state.insert(0, event)
            paddings.append(event)
        else:
            # padding_front_state.append(event)
            padding_end_state.append(event)

    padding_end_state = padding_end_state + paddings
    return padding_end_state


def make_csv_data_line(state_features, state_index):
    # flat the state features of all histories
    state_feature_str = ''
    for history_index in range(0, len(state_features[state_index])):
        # for history_index in range(0, 1): # only consider the curent state
        for feature_index in range(0, len(state_features[state_index][history_index])):
            # ignore actions of current state, since we only generate data for 1 action
            if history_index == 9 and feature_index >= 12:
                continue

            state_feature_value = state_features[state_index][history_index][feature_index]

            # action is no longer normalized
            # # check if it is action and change action to one-hot
            # if feature_index >= 12: 
            #     if state_features[state_index][history_index][feature_index] > 0:
            #         state_feature_value = 1
            #     else:
            #         state_feature_value = 0

            state_feature_str = state_feature_str + str(state_feature_value).strip() + ','

    return state_feature_str


def write_Q_data_txt(Q_file_Writer, Q_values, padding_front_states):
    # Q values, trace lengths, and states are created in the same order, so 'state_index' works for all of them
    for state_index in range(0, len(Q_values)):

        # generate the data only if the action of the current event is what we want
        # current event is index 9
        # if padding_front_states[state_index][9][action_index_in_feature] > 0:

        Q_value = str(Q_values[state_index][0]).strip()  # only the Q_home for now, [0]
        state_feature_str = make_csv_data_line(padding_front_states, state_index)

        # write a line [Q, state_features_history_1, state_features_history_2, one_hot_action_history_2, ..., state_features_history_10, one_hot_action_history_10]
        # [:-1] to remove last comma
        if Q_file_Writer is not None:
            Q_file_Writer.write(Q_value.strip() + ',' + (state_feature_str.strip()[:-1]) + '\n')


def generate(sess, model, Q_file_Writer):
    # loading network
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, SAVED_NETWORK + trained_model_name)
    print 'successfully load data from' + SAVED_NETWORK

    # generate data with Q vaules using data from all subfolders in processed data folder
    for dir_game in DIR_GAMES_ALL:

        # skip the hidden mac file, feature_var.txt and feature_mean.txt
        if 'DS_Store' in dir_game or "feature_var" in dir_game or "feature_mean" in dir_game:
            continue

        print("\n loadeding files in folder " + str(dir_game) + '...')

        # find data file names
        reward_name = None
        state_input_name = None
        action_input_name = None
        state_trace_length_name = None
        game_files = os.listdir(DATA_STORE + "/" + dir_game)
        for filename in game_files:
            if "reward" in filename:
                reward_name = filename
            elif "state_feature_seq" in filename:
                state_input_name = filename
            elif "action_feature_seq" in filename:
                action_input_name = filename
            elif "lt" in filename:
                state_trace_length_name = filename

        if reward_name is None or state_input_name is None or action_input_name is None or state_trace_length_name is None:
            print("\n skip folder " + str(dir_game) + ' because of missing files')
            continue

        reward = sio.loadmat(DATA_STORE + "/" + dir_game + "/" + reward_name)
        reward = reward['reward'][0]

        state_input = sio.loadmat(DATA_STORE + "/" + dir_game + "/" + state_input_name)
        state_input = (state_input['state_feature_seq'])

        action_input = sio.loadmat(DATA_STORE + "/" + dir_game + "/" + action_input_name)
        action_input = (action_input['action_feature_seq'])

        state_trace_length = sio.loadmat(DATA_STORE + "/" + dir_game + "/" + state_trace_length_name)
        state_trace_length = (state_trace_length['lt'])[0]

        print("\n loaded files in folder " + str(dir_game) + " successfully")

        if len(state_input) != len(reward) or len(action_input) != len(reward) or len(state_trace_length) != len(
                reward):
            raise Exception('state/action length does not equal to reward length')

        train_len = len(state_input)
        train_number = 0
        # state representation is [state_features, one-hot-action]
        s_t0 = np.concatenate((state_input[train_number], action_input[train_number]), axis=1)
        train_number += 1

        # action_index_in_feature = get_action_index_in_feature(action_index)

        Q_values_game = []
        padding_front_states_game = []
        # impacts_game = []
        trace_t0_game = []

        while True:
            batch_return, train_number, s_tl = get_together_training_batch(s_t0,
                                                                           state_input, action_input, reward,
                                                                           train_number, train_len, state_trace_length,
                                                                           BATCH_SIZE)

            # get the batch variables
            s_t0_batch = [d[0] for d in batch_return]
            trace_t0_batch = [d[3] for d in batch_return]

            terminal = batch_return[len(batch_return) - 1][5]

            # calculate Q values 
            # home Q values for both home taking possession and away taking possession
            [Q_values] = sess.run([model.read_out],
                                  feed_dict={model.rnn_input_ph: s_t0_batch, model.trace_lengths_ph: trace_t0_batch})
            Q_values_game.extend(Q_values)
            trace_t0_game.extend(trace_t0_batch)

            # move padding events from the end to the front
            for padding_end_state in s_t0_batch:
                padding_front_state = convert_to_padding_front(padding_end_state)
                padding_front_states_game.append(padding_front_state)

            s_t0 = s_tl

            if terminal:
                break

        # write data for a whole game 
        write_Q_data_txt(Q_file_Writer, Q_values_game, padding_front_states_game)


def generation_start(Q_file_Writer):
    sess_nn = tf.InteractiveSession()

    # define model 
    icehockey_config_path = os.path.dirname(
        os.path.realpath(__file__)) + "/../environment_settings/ice_hockey_predict_Qs_lstm.yaml"
    icehockey_model_config = LSTMQsCongfig.load(icehockey_config_path)
    model_nn = TD_Prediction(config=icehockey_model_config)
    model_nn()
    sess_nn.run(tf.global_variables_initializer())

    generate(sess_nn, model_nn, Q_file_Writer)


# value_type==1: Q
# value_type==2: impact
def generete_csv_header(file_Writer, value_type):
    # 3: data file name, NA, which line to start with
    # 1: Q
    # (12 + 27) * 9: (state features + one hot action) * 9 history events
    # 12: the state features of 1st event, ignore actions
    header_str = ''
    history_count = 10
    for line in range(0, 3 + 1 + (12 + 27) * 9 + 12):
        if line == 0:
            # Q_file_Writer.write(Q_file_name + '\n')
            pass
        elif line == 1:
            # Q_file_Writer.write('NA\n')
            pass
        elif line == 2:
            # Q_file_Writer.write('1\n')
            pass
        elif line == 3:
            if value_type == 1:
                header_str = header_str + 'Q,'
            elif value_type == 2:
                header_str = header_str + 'impact,'
            else:
                raise ValueError('Q or impact??? must be one of them.')

        elif line == 4 or (line - 3 - 1) % 39 == 0:
            history_count = history_count - 1
            header_str = header_str + 'xAdjCoord' + str(history_count) + ','

        elif line == 5 or (line - 3 - 1) % 39 == 1:
            header_str = header_str + 'yAdjCoord' + str(history_count) + ','

        elif line == 6 or (line - 3 - 1) % 39 == 2:
            header_str = header_str + 'scoreDifferential' + str(history_count) + ','

        elif line == 7 or (line - 3 - 1) % 39 == 3:
            header_str = header_str + 'manpowerSituation' + str(history_count) + ','

        elif line == 8 or (line - 3 - 1) % 39 == 4:
            header_str = header_str + 'outcome' + str(history_count) + ','

        elif line == 9 or (line - 3 - 1) % 39 == 5:
            header_str = header_str + 'velocity_x' + str(history_count) + ','

        elif line == 10 or (line - 3 - 1) % 39 == 6:
            header_str = header_str + 'velocity_y' + str(history_count) + ','

        elif line == 11 or (line - 3 - 1) % 39 == 7:
            header_str = header_str + 'time_remain' + str(history_count) + ','

        elif line == 12 or (line - 3 - 1) % 39 == 8:
            header_str = header_str + 'duration' + str(history_count) + ','

        elif line == 13 or (line - 3 - 1) % 39 == 9:
            header_str = header_str + 'home' + str(history_count) + ','

        elif line == 14 or (line - 3 - 1) % 39 == 10:
            header_str = header_str + 'away' + str(history_count) + ','

        elif line == 15 or (line - 3 - 1) % 39 == 11:
            header_str = header_str + 'angle2gate' + str(history_count) + ','

        else:  # actions
            index = (line - 3 - 1) % 39 - 12
            action = action_all[index]
            header_str = header_str + action + str(history_count) + ','

    # [:-1] to remove last comma
    header_str = header_str[:-1]
    file_Writer.write(header_str + '\n')


if __name__ == '__main__':
    if not os.path.isdir(Q_data_DIR):
        os.mkdir(Q_data_DIR)

    Q_file_Writer = open(Q_data_DIR + '/' + Q_file_name, 'w')

    if Q_file_Writer is not None:
        generete_csv_header(Q_file_Writer, 1)  # value_type==1: Q

    # the generated Q data file only contains data which has action 'ACTION_TO_MIMIC'
    # action_index = action_all.index(ACTION_TO_MIMIC)

    generation_start(Q_file_Writer)

    if Q_file_Writer is not None:
        Q_file_Writer.close()
