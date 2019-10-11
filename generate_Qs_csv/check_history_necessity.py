import datetime
import tensorflow as tf
import os
import scipy.io as sio
import numpy as np
import copy

import sys
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + "/..")

from nn_structure.cvrnn import CVRNN
from nn_structure.lstm_Qs_nn import TD_Prediction
from nn_structure.lstm_score_diff_nn import Diff_Prediction

from config.LSTM_Qs_config import LSTMQsCongfig
from config.cvrnn_config import CVRNNCongfig

from utils import handle_trace_length, get_together_training_batch, compromise_state_trace_length
from configuration import MODEL_TYPE, MAX_TRACE_LENGTH, FEATURE_NUMBER, BATCH_SIZE, GAMMA, H_SIZE, \
    model_train_continue, FEATURE_TYPE, ITERATE_NUM, learning_rate, SPORT, directory_generated_Q_data, \
    save_mother_dir, action_all

ACTION_TO_MIMIC = 'shot'

trained_model_name = 'Ice-Hockey-game--901'

SAVED_NETWORK = save_mother_dir + '/models/'

Q_data_DIR = save_mother_dir + '/' + directory_generated_Q_data

DATA_STORE = "/cs/oschulte/xiangyus/2019-icehockey-data-preprocessed/2018-2019"
# DATA_STORE = "/Users/xiangyusun/Desktop/2019-icehockey-data-preprocessed/2018-2019"

DIR_GAMES_ALL = os.listdir(DATA_STORE)

timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
Q_file_name = 'Q_' + ACTION_TO_MIMIC + '_' + str(timestamp) + '.csv'
impact_file_name = 'impact_' + ACTION_TO_MIMIC + '_' + str(timestamp) + '.csv'

def write_Q_data_txt(fileWriter, Q_values, Q_values_no_history_game, state_features, action_index, max_abs_diff):
    for state_index in range(0, len(Q_values)):
        Q_value = Q_values[state_index][0] # only the Q_home for now, [0]
        Q_value_no_history = Q_values_no_history_game[state_index][0]
        difference = Q_value - Q_value_no_history

        if abs(difference) > max_abs_diff:
            max_abs_diff = abs(difference)

        # the first 12 elements are state features
        action_index_in_feature = 12 + action_index

        # generate the data only if the action of the current event is what we want
        # current event is index 9
        if state_features[state_index][9][action_index_in_feature] > 0:
            fileWriter.write(str(Q_value).strip() + ',' + str(Q_value_no_history).strip() + ',' + str(difference).strip() + ',' + '\n')
    
    return max_abs_diff

def generate(sess, model, fileWriter, action_index):
    # loading network
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, SAVED_NETWORK + trained_model_name)
    print 'successfully load data from' + SAVED_NETWORK

    max_abs_diff = 0 

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

        if len(state_input) != len(reward) or len(action_input) != len(reward) or len(state_trace_length) != len(reward):
                raise Exception('state/action length does not equal to reward length')

        train_len = len(state_input)
        train_number = 0
        # state representation is [state_features, one-hot-action]
        s_t0 = np.concatenate((state_input[train_number], action_input[train_number]), axis=1)
        train_number += 1

        Q_values_game = []
        Q_values_no_history_game = []
        padding_front_states_game = []

        while True:
            batch_return, train_number, s_tl = get_together_training_batch(s_t0,
            state_input,action_input,reward,train_number,train_len,state_trace_length,BATCH_SIZE)

            # get the batch variables
            s_t0_batch = [d[0] for d in batch_return]
            trace_t0_batch = [d[3] for d in batch_return]

            terminal = batch_return[len(batch_return) - 1][5]

            # calculate Q values 
            # home Q values for both home taking possession and away taking possession
            [Q_values] = sess.run([model.read_out], feed_dict={model.rnn_input_ph: s_t0_batch, model.trace_lengths_ph: trace_t0_batch})
            Q_values_game.extend(Q_values)

            s_t0_batch_no_history = copy.deepcopy(s_t0_batch)

            # calculate Q values for which all history variables set to 0
            for batch_no_history in s_t0_batch_no_history:
                for event_number in range(0, len(batch_no_history)):
                    if event_number > 0:
                        batch_no_history[event_number] = [0] * 39 # 12 features + 27 one hot action

            [Q_values_no_history] = sess.run([model.read_out], feed_dict={model.rnn_input_ph: s_t0_batch_no_history, model.trace_lengths_ph: trace_t0_batch})
            Q_values_no_history_game.extend(Q_values_no_history)

            # move padding events from the end to the front
            for state in s_t0_batch:
                padding_front_state = []
                for event in state:
                    # if both event home and away are 0, it's paddings
                    home = event[9]
                    away = event[10]
                    if home == 0 and away == 0:
                        # the event is padding
                        padding_front_state.insert(0, event)
                    else:
                        padding_front_state.append(event)

                padding_front_states_game.append(padding_front_state)

            s_t0 = s_tl

            if terminal:
                break

        # write data for a whole game 
        max_abs_diff = write_Q_data_txt(fileWriter, Q_values_game, Q_values_no_history_game, padding_front_states_game, action_index, max_abs_diff)

    fileWriter.write(' , , max: ' + str(max_abs_diff) + '\n')

def generation_start(fileWriter, action_index):
    sess_nn = tf.InteractiveSession()

    # define model 
    icehockey_config_path = os.path.dirname(os.path.realpath(__file__)) + "/../environment_settings/ice_hockey_predict_Qs_lstm.yaml"
    icehockey_model_config = LSTMQsCongfig.load(icehockey_config_path)
    model_nn = TD_Prediction(config=icehockey_model_config)
    model_nn()
    sess_nn.run(tf.global_variables_initializer())

    generate(sess_nn, model_nn, fileWriter, action_index)

def generete_csv_header(file_Writer):
    header_str = 'Q, Q_no_history, Differenct'
    file_Writer.write(header_str + '\n')

if __name__ == '__main__':
    if not os.path.isdir(Q_data_DIR):
        os.mkdir(Q_data_DIR)

    fileWriter = open(Q_data_DIR + '/' + Q_file_name, 'w')

    generete_csv_header(fileWriter)

    # the generated Q data file only contains data which has action 'ACTION_TO_MIMIC'
    action_index = action_all.index(ACTION_TO_MIMIC)
    
    generation_start(fileWriter, action_index)
