import numpy as np
import os
import csv
import math
import json
import scipy.io as sio
import unicodedata


# from config.icehockey_feature_setting import select_feature_setting


def handle_trace_length(state_trace_length):
    """
    transform format of trace length
    :return:
    """
    trace_length_record = []
    for length in state_trace_length:
        for sub_length in range(0, int(length)):
            trace_length_record.append(sub_length + 1)

    return trace_length_record


def compromise_state_trace_length(state_trace_length, state_input, reward, max_trace_length, features_num):
    state_trace_length_output = []
    for index in range(0, len(state_trace_length)):
        tl = state_trace_length[index]
        if tl >= 10:
            tl = 10
        if tl > max_trace_length:
            state_input_change_list = []
            state_input_org = state_input[index]
            reward_change_list = []
            reward_org = reward[index]
            for i in range(0, max_trace_length):
                state_input_change_list.append(state_input_org[tl - max_trace_length + i])
                temp = reward_org[tl - max_trace_length + i]
                # if temp != 0:
                #     print 'find miss reward'
                reward_change_list.append(reward_org[tl - max_trace_length + i])

            state_input_update = padding_hybrid_feature_input(state_input_change_list,
                                                              max_trace_length=max_trace_length,
                                                              features_num=features_num)
            state_input[index] = state_input_update
            reward_update = padding_hybrid_reward(reward_change_list)
            reward[index] = reward_update

            tl = max_trace_length
        state_trace_length_output.append(tl)
    return state_trace_length_output, state_input, reward


# def padding_hybrid_feature_input(hybrid_feature_input):
#     current_list_length = len(hybrid_feature_input)
#     padding_list_length = 10 - current_list_length
#     for i in range(0, padding_list_length):
#         hybrid_feature_input.append(np.asarray([float(0)] * 25))
#     return np.asarray(hybrid_feature_input)


def padding_hybrid_reward(hybrid_reward):
    current_list_length = len(hybrid_reward)
    padding_list_length = 10 - current_list_length
    for i in range(0, padding_list_length):
        hybrid_reward.append(0)
    return np.asarray(hybrid_reward)


def get_soccer_game_data(data_store, dir_game, config):
    game_files = os.listdir(data_store + "/" + dir_game)
    for filename in game_files:
        if "reward" in filename:
            reward_name = filename
        elif "state" in filename:
            state_input_name = filename
        elif "trace" in filename:
            state_trace_length_name = filename
        elif "home_away" in filename:
            ha_id_name = filename

    reward = sio.loadmat(data_store + "/" + dir_game + "/" + reward_name)['reward']
    state_input = sio.loadmat(data_store + "/" + dir_game + "/" + state_input_name)['state']
    ha_id = sio.loadmat(data_store + "/" + dir_game + "/" + ha_id_name)["home_away"][0].astype(int)
    # state_input = (state_input['dynamic_feature_input'])
    # state_output = sio.loadmat(DATA_STORE + "/" + dir_game + "/" + state_output_name)
    # state_output = state_output['hybrid_output_state']
    state_trace_length = sio.loadmat(
        data_store + "/" + dir_game + "/" + state_trace_length_name)['trace_length'][0]
    # state_trace_length = (state_trace_length['hybrid_trace_length'])[0]
    state_trace_length = handle_trace_length(state_trace_length)
    state_trace_length, state_input, reward = compromise_state_trace_length(
        state_trace_length=state_trace_length,
        state_input=state_input,
        reward=reward,
        max_trace_length=config.learn.max_trace_length,
        features_num=config.learn.feature_number
    )
    return state_trace_length, state_input, reward, ha_id


def get_icehockey_game_data_old(data_store, dir_game, config):
    game_files = os.listdir(data_store + "/" + dir_game)
    for filename in game_files:
        if "dynamic_rnn_reward" in filename:
            reward_name = filename
        elif "dynamic_rnn_input" in filename:
            state_input_name = filename
        elif "trace" in filename:
            state_trace_length_name = filename
        elif "home_identifier" in filename:
            ha_id_name = filename
        elif 'team_id' in filename:
            team_id_name = filename

    reward = sio.loadmat(data_store + "/" + dir_game + "/" + reward_name)
    ha_id = sio.loadmat(data_store + "/" + dir_game + "/" + ha_id_name)["home_identifier"][0]
    team_id = sio.loadmat(data_store + "/" + dir_game + "/" + team_id_name)["team_id"][0]
    try:
        reward = reward['dynamic_rnn_reward']
    except:
        print "\n" + dir_game
        raise ValueError("reward wrong")
    state_input = sio.loadmat(data_store + "/" + dir_game + "/" + state_input_name)['dynamic_feature_input']
    # state_input = (state_input['dynamic_feature_input'])
    # state_output = sio.loadmat(DATA_STORE + "/" + dir_game + "/" + state_output_name)
    # state_output = state_output['hybrid_output_state']
    state_trace_length = sio.loadmat(
        data_store + "/" + dir_game + "/" + state_trace_length_name)['hybrid_trace_length'][0]
    # state_trace_length = (state_trace_length['hybrid_trace_length'])[0]
    state_trace_length = handle_trace_length(state_trace_length)
    state_trace_length, state_input, reward = compromise_state_trace_length(
        state_trace_length=state_trace_length,
        state_input=state_input,
        reward=reward,
        max_trace_length=config.learn.max_trace_length,
        features_num=config.learn.feature_number
    )

    return state_trace_length, state_input, reward, ha_id, team_id


def transfer2seq(data, trace_length, max_length):
    return_data = []
    for index in range(0, len(trace_length)):
        tl = trace_length[index]
        # print(index)
        tl = max_length if tl > max_length else tl
        seq_line = data[index:index + tl].tolist()
        for i in range(max_length - len(seq_line)):
            seq_line.append(len(data[0]) * [0])
        # print len(seq_line)
        assert len(seq_line) == max_length
        return_data.append(seq_line)
    return np.asarray(return_data)


def generate_selection_matrix(trace_lengths, max_trace_length):
    selection_matrix = []
    for trace_length in trace_lengths:
        selection_matrix.append(trace_length * [1] + (max_trace_length - trace_length) * [0])
    return np.asarray(selection_matrix)


def get_icehockey_game_data(data_store, dir_game, config):
    game_files = os.listdir(data_store + "/" + dir_game)
    reward_name = None
    state_input_name = None
    trace_length_name = None
    # ha_id_name = None
    team_id_name = None
    action_id_name = None
    player_index_name = None

    for filename in game_files:
        if "reward" in filename:
            reward_name = filename
        elif "state_feature_seq" in filename:
            state_input_name = filename
        elif "lt" in filename:
            trace_length_name = filename
        # elif "home_identifier" in filename:
        #     ha_id_name = filename
        elif 'team' in filename:
            team_id_name = filename
        elif 'player_index' in filename:
            player_index_name = filename
        elif 'action' in filename:
            if 'action_feature_seq' in filename:
                continue
            action_id_name = filename
    if reward_name is not None:
        reward = sio.loadmat(data_store + "/" + dir_game + "/" + reward_name)
        reward = reward['reward'][0]
    # if ha_id_name is not None:
    #     ha_id = sio.loadmat(data_store + "/" + dir_game + "/" + ha_id_name)["home_identifier"][0]
    if team_id_name is not None:
        team_id = sio.loadmat(data_store + "/" + dir_game + "/" + team_id_name)['team']
    if state_input_name is not None:
        state_input = sio.loadmat(data_store + "/" + dir_game + "/" + state_input_name)['state_feature_seq']
    if action_id_name is not None:
        action = sio.loadmat(data_store + "/" + dir_game + "/" + action_id_name)['action']
    if player_index_name is not None:
        player_index = sio.loadmat(data_store + "/" + dir_game + "/" + player_index_name)['player_index']
        # state_input = (state_input['dynamic_feature_input'])
    # state_output = sio.loadmat(DATA_STORE + "/" + dir_game + "/" + state_output_name)
    # state_output = state_output['hybrid_output_state']
    if trace_length_name is not None:
        state_trace_length = sio.loadmat(
            data_store + "/" + dir_game + "/" + trace_length_name)['lt'][0]

    return state_trace_length, state_input, reward, action, team_id, player_index


def id2onehot(id, dimension_num):
    onehot = [0] * dimension_num
    onehot[id] = 1
    return onehot


def get_together_training_batch(s_t0, state_input, reward, player_index, train_number, train_len, state_trace_length,
                                action,
                                team_id, config):
    """
    combine training data to a batch
    :return:
    """
    batch_size = config.Learn.batch_size
    batch_return = []
    print_flag = False
    current_batch_length = 0
    while current_batch_length < batch_size:
        s_t1 = state_input[train_number]
        if len(s_t1) < 10 or len(s_t0) < 10:
            raise ValueError("wrong length of s")
            # train_number += 1
            # continue
        s_length_t1 = state_trace_length[train_number]
        s_length_t0 = state_trace_length[train_number - 1]
        action_id_t1 = action[train_number]
        action_id_t0 = action[train_number - 1]
        team_id_t1 = team_id[train_number]
        team_id_t0 = team_id[train_number - 1]
        player_index_t1 = player_index[train_number]
        player_index_t0 = player_index[train_number - 1]
        # team_id_t1 = id2onehot(team_id[train_number], config.learn.team_number)
        # team_id_t0 = id2onehot(team_id[train_number - 1], config.learn.team_number)
        if s_length_t1 > 10:  # if trace length is too long
            s_length_t1 = 10
        if s_length_t0 > 10:  # if trace length is too long
            s_length_t0 = 10
        try:
            s_reward_t1 = reward[train_number]
            s_reward_t0 = reward[train_number - 1]
        except IndexError:
            raise IndexError("s_reward wrong with index")
        train_number += 1
        if train_number + 1 == train_len:
            r_t0 = s_reward_t0
            r_t1 = s_reward_t1
            if r_t0 == float(0):
                r_t0_combine = [float(0), float(0), float(0)]
                batch_return.append(
                    (s_t0, s_t1, r_t0_combine, s_length_t0, s_length_t1, action_id_t0, action_id_t1, team_id_t0,
                     team_id_t1, player_index_t0, player_index_t1, 0, 0))
                if r_t1 == float(0):
                    r_t1_combine = [float(0), float(0), float(1)]
                elif r_t1 == float(-1):
                    r_t1_combine = [float(0), float(1), float(1)]
                elif r_t1 == float(1):
                    r_t1_combine = [float(1), float(0), float(1)]
                else:
                    raise ValueError("incorrect r_t1")
                batch_return.append(
                    (s_t1, s_t1, r_t1_combine, s_length_t1, s_length_t1, action_id_t1, action_id_t1, team_id_t1,
                     team_id_t1, player_index_t0, player_index_t1, 1, 0))

            elif r_t0 == float(-1):
                r_t0_combine = [float(0), float(1), float(0)]
                batch_return.append(
                    (s_t0, s_t1, r_t0_combine, s_length_t0, s_length_t1, action_id_t0, action_id_t1, team_id_t0,
                     team_id_t1, player_index_t0, player_index_t1, 0, 0))
                if r_t1 == float(0):
                    r_t1_combine = [float(0), float(0), float(1)]
                elif r_t1 == float(-1):
                    r_t1_combine = [float(0), float(1), float(1)]
                elif r_t1 == float(1):
                    r_t1_combine = [float(1), float(0), float(1)]
                else:
                    raise ValueError("incorrect r_t1")
                batch_return.append(
                    (s_t1, s_t1, r_t1_combine, s_length_t1, s_length_t1, action_id_t1, action_id_t1, team_id_t1,
                     team_id_t1, player_index_t0, player_index_t1, 1, 0))

            elif r_t0 == float(1):
                r_t0_combine = [float(1), float(0), float(0)]
                batch_return.append(
                    (s_t0, s_t1, r_t0_combine, s_length_t0, s_length_t1, action_id_t0, action_id_t1, team_id_t0,
                     team_id_t1, player_index_t0, player_index_t1, 0, 0))

                if r_t1 == float(0):
                    r_t1_combine = [float(0), float(0), float(1)]
                elif r_t1 == float(-1):
                    r_t1_combine = [float(0), float(1), float(1)]
                elif r_t1 == float(1):
                    r_t1_combine = [float(1), float(0), float(1)]
                else:
                    raise ValueError("incorrect r_t1")
                batch_return.append(
                    (s_t1, s_t1, r_t1_combine, s_length_t1, s_length_t1, action_id_t1, action_id_t1, team_id_t1,
                     team_id_t1, player_index_t0, player_index_t1, 1, 0))
            else:
                raise ValueError("r_t0 wrong value")

            s_t0 = s_t1
            break

        r_t0 = s_reward_t0
        if r_t0 != float(0):
            # print 'find no-zero reward', r_t0
            print_flag = True
            if r_t0 == [float(-1)]:
                r_t0_combine = [float(0), float(1), float(0)]
                batch_return.append(
                    (s_t0, s_t1, r_t0_combine, s_length_t0, s_length_t1, action_id_t0, action_id_t1, team_id_t0,
                     team_id_t1, player_index_t0, player_index_t1, 0, 1))
            elif r_t0 == [float(1)]:
                r_t0_combine = [float(1), float(0), float(0)]
                batch_return.append(
                    (s_t0, s_t1, r_t0_combine, s_length_t0, s_length_t1, action_id_t0, action_id_t1, team_id_t0,
                     team_id_t1, player_index_t0, player_index_t1, 0, 1))
            else:
                raise ValueError("r_t0 wrong value")
            s_t0 = s_t1
            break
        r_t0_combine = [float(0), float(0), float(0)]
        batch_return.append(
            (s_t0, s_t1, r_t0_combine, s_length_t0, s_length_t1, action_id_t0, action_id_t1, team_id_t0,
             team_id_t1, player_index_t0, player_index_t1, 0, 0))
        current_batch_length += 1
        s_t0 = s_t1

    return batch_return, train_number, s_t0, print_flag


def write_game_average_csv(data_record, log_dir):
    if os.path.exists(log_dir + '/avg_cost_record.csv'):
        with open(log_dir + '/avg_cost_record.csv', 'a') as csvfile:
            fieldnames = (data_record[0]).keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            for record in data_record:
                writer.writerow(record)
    else:
        with open(log_dir + '/avg_cost_record.csv', 'w') as csvfile:
            fieldnames = (data_record[0]).keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for record in data_record:
                writer.writerow(record)


def judge_feature_in_action(feature_input, actions):
    for action in actions:
        if feature_input == action:
            return True
    return False


def construct_simulation_data(features_train, features_mean, features_scale,
                              feature_type, is_home, action_type, actions, set_dict={}):
    state = []
    for feature in features_train:
        if feature == 'xAdjCoord':
            xAdjCoord = set_dict.get('xAdjCoord')
            scale_xAdjCoord = float(xAdjCoord - features_mean['xAdjCoord']) / features_scale['xAdjCoord']
            state.append(scale_xAdjCoord)
        elif feature == 'yAdjCoord':
            yAdjCoord = set_dict.get('yAdjCoord')
            scale_yAdjCoord = float(yAdjCoord - features_mean['yAdjCoord']) / features_scale['yAdjCoord']
            state.append(scale_yAdjCoord)
        elif feature in set_dict:
            temp = set_dict[feature]
            scale_temp = float(temp - features_mean[feature]) / features_scale[feature]
            state.append(scale_temp)
        elif feature_type < 5 and feature == 'event_id':
            actions = {'block': 0,
                       'carry': 1,
                       'check': 2,
                       'dumpin': 3,
                       'dumpout': 4,
                       'goal': 5,
                       'lpr': 6,
                       'offside': 7,
                       'pass': 8,
                       'puckprotection': 9,
                       'reception': 10,
                       'shot': 11,
                       'shotagainst': 12}
            scale_actions = float(actions[action_type] - features_mean['event_id']) / features_scale['event_id']
            state.append(scale_actions)

        elif feature_type >= 5 and action_type == feature:
            scale_action = float(1 - features_mean[action_type]) / features_scale[action_type]
            state.append(scale_action)
        elif feature_type >= 5 and judge_feature_in_action(feature, actions):
            scale_action = float(0 - features_mean[feature]) / features_scale[feature]
            state.append(scale_action)
        elif feature == 'event_outcome':
            scale_event_outcome = float(1 - features_mean['event_outcome']) / features_scale['event_outcome']
            state.append(scale_event_outcome)
        elif feature == 'angel2gate':
            gate_x_coord = 89
            gate_y_coord = 0
            y_diff = abs(yAdjCoord - gate_y_coord)
            x_diff = gate_x_coord - xAdjCoord
            z = math.sqrt(math.pow(y_diff, 2) + math.pow(x_diff, 2))
            try:
                angel2gate = math.acos(float(x_diff) / z)
            except:
                print ("exception point with x:{0} and y:{1}".format(str(xAdjCoord), str(yAdjCoord)))
                angel2gate = math.pi
            scale_angel2gate = float(angel2gate - features_mean['angel2gate']) / features_scale['angel2gate']
            state.append(scale_angel2gate)
        elif feature == 'home':
            if is_home:
                scale_home = float(1 - features_mean['home']) / features_scale['home']
                state.append(scale_home)
            else:
                scale_home = float(0 - features_mean['home']) / features_scale['home']
                state.append(scale_home)
        elif feature == 'away':
            if is_home:
                scale_away = float(0 - features_mean['away']) / features_scale['away']
                state.append(scale_away)
            else:
                scale_away = float(1 - features_mean['away']) / features_scale['away']
                state.append(scale_away)
        else:
            state.append(0)

    return np.asarray(state)


def padding_hybrid_feature_input(hybrid_feature_input, max_trace_length, features_num):
    current_list_length = len(hybrid_feature_input)
    padding_list_length = max_trace_length - current_list_length
    for i in range(0, padding_list_length):
        hybrid_feature_input.append(np.asarray([float(0)] * features_num))
    return hybrid_feature_input


def start_lstm_generate_spatial_simulation(history_action_type, history_action_type_coord,
                                           action_type, data_simulation_dir, simulation_type,
                                           feature_type, max_trace_length, features_num, is_home=True):
    simulated_data_all = []

    features_train, features_mean, features_scale, actions = select_feature_setting(feature_type=feature_type)

    for history_index in range(0, len(history_action_type) + 1):
        state_ycoord_list = []
        for ycoord in np.linspace(-42.5, 42.5, 171):
            state_xcoord_list = []
            for xcoord in np.linspace(-100.0, 100.0, 401):
                set_dict = {'xAdjCoord': xcoord, 'yAdjCoord': ycoord}
                state_generated = construct_simulation_data(
                    features_train=features_train,
                    features_mean=features_mean,
                    features_scale=features_scale,
                    feature_type=feature_type,
                    is_home=is_home,
                    action_type=action_type,
                    actions=actions,
                    set_dict=set_dict)
                state_generated_list = [state_generated]
                for inner_history in range(0, history_index):
                    xAdjCoord = history_action_type_coord[inner_history].get('xAdjCoord')
                    yAdjCoord = history_action_type_coord[inner_history].get('yAdjCoord')
                    action = history_action_type[inner_history]
                    if action != action_type:
                        set_dict_history = {'xAdjCoord': xAdjCoord, 'yAdjCoord': yAdjCoord, action: 1, action_type: 0}
                    else:
                        set_dict_history = {'xAdjCoord': xAdjCoord, 'yAdjCoord': yAdjCoord, action: 1}
                    state_generated_history = construct_simulation_data(
                        features_train=features_train,
                        features_mean=features_mean,
                        features_scale=features_scale,
                        feature_type=feature_type,
                        is_home=is_home,
                        action_type=action_type,
                        actions=actions,
                        set_dict=set_dict_history, )
                    state_generated_list = [state_generated_history] + state_generated_list

                state_generated_padding = padding_hybrid_feature_input(
                    hybrid_feature_input=state_generated_list,
                    max_trace_length=max_trace_length,
                    features_num=features_num)
                state_xcoord_list.append(state_generated_padding)
            state_ycoord_list.append(np.asarray(state_xcoord_list))

        store_data_dir = data_simulation_dir + '/' + simulation_type

        if not os.path.isdir(store_data_dir):
            os.makedirs(store_data_dir)
        # else:
        #     raise Exception
        if is_home:
            sio.savemat(
                store_data_dir + "/LSTM_Home_" + simulation_type + "-" + action_type + '-' + str(
                    history_action_type[0:history_index]) + "-feature" + str(
                    feature_type) + ".mat",
                {'simulate_data': np.asarray(state_ycoord_list)})
        else:
            sio.savemat(
                store_data_dir + "/LSTM_Away_" + simulation_type + "-" + action_type + '-' + str(
                    history_action_type[0:history_index]) + "-feature" + str(
                    feature_type) + ".mat",
                {'simulate_data': np.asarray(state_ycoord_list)})
        simulated_data_all.append(np.asarray(state_ycoord_list))

    return simulated_data_all


def find_game_dir(dir_all, data_path, target_game_id, sports='IceHockey'):
    if sports == 'IceHockey':
        game_name = None
        for directory in dir_all:
            game = sio.loadmat(data_path + "/" + str(directory))
            gameId = (game['x'])['gameId'][0][0][0]
            gameId = unicodedata.normalize('NFKD', gameId).encode('ascii', 'ignore')
            if gameId == target_game_id:
                game_name = directory
                print directory
                break
    elif sports == 'Soccer':
        for directory in dir_all:
            with open(data_path + str(directory)) as f:
                data = json.load(f)[0]
            gameId = str(data.get('gameId'))
            # gameId = unicodedata.normalize('NFKD', gameId).encode('ascii', 'ignore')
            print str(data.get('gameDate'))
            # if gameId == target_game_id:
            #     game_name = directory
            #     print directory
            #     break
    else:
        raise ValueError('Unknown sports game')

    if game_name:
        return game_name.split(".")[0]
    else:
        raise ValueError("can't find the game {0}".format(str(target_game_id)))


def normalize_data(game_value_home, game_value_away, game_value_end):
    game_value_home_normalized = []
    game_value_away_normalized = []
    game_value_end_normalized = []
    for index in range(0, len(game_value_home)):
        home_value = game_value_home[index]
        away_value = game_value_away[index]
        end_value = game_value_end[index]
        if end_value < 0:
            end_value = 0
        if away_value < 0:
            away_value = 0
        if home_value < 0:
            home_value = 0
        game_value_home_normalized.append(float(home_value) / (home_value + away_value + end_value))
        game_value_away_normalized.append(float(away_value) / (home_value + away_value + end_value))
        game_value_end_normalized.append(float(end_value) / (home_value + away_value + end_value))
    return np.asarray(game_value_home_normalized), np.asarray(game_value_away_normalized), np.asarray(
        game_value_end_normalized)


def read_player_stats(player_scoring_stats_dir):
    with open(player_scoring_stats_dir, 'r') as f:
        data_all = f.readlines()
    return data_all


def match_player_name_id(player_basic_info, player_scoring_stats):
    from resource.ice_hockey_201819.player_name_matching_dir import name_matching, team_matching
    # TODO: fix 3 duplicated name with team name, but player might transfer team
    for id in player_basic_info.keys():
        player_first_name_basic = player_basic_info.get(id).get('first_name')
        player_last_name_basic = player_basic_info.get(id).get('last_name')
        player_team_basic = player_basic_info.get(id).get('teamName')

        if name_matching.get(player_first_name_basic + '|' + player_last_name_basic):
            player_name_basic_match = name_matching.get(player_first_name_basic + '|' + player_last_name_basic)
            player_first_name_basic = player_name_basic_match.split('|')[0]
            player_last_name_basic = player_name_basic_match.split('|')[1]
        if team_matching.get(player_team_basic):
            player_team_basic = team_matching.get(player_team_basic)
        match_id = None
        match_name = None
        possible_name_list = []
        for player_scoring_stat in player_scoring_stats[1:]:
            player_scoring_stat_split = player_scoring_stat.split(',')
            player_first_name_score = player_scoring_stat_split[1].replace('"', '')
            player_last_name_score = player_scoring_stat_split[0].replace('"', '')
            player_team_score = player_scoring_stat_split[2]
            if player_first_name_basic == player_first_name_score and player_last_name_basic == player_last_name_score:
                # and player_team_basic==player_team_score:
                match_id = id
                match_name = player_first_name_score + ' ' + player_last_name_score
                break
            if player_first_name_basic == player_first_name_score or player_last_name_basic == player_last_name_score:
                possible_name_list.append(player_first_name_score + '|' + player_last_name_score)

        # print match_name
        if match_name is None:
            print player_first_name_basic + '|' + player_last_name_basic + ' possible:' + str(possible_name_list)


def check_duplicate_name(player_scoring_stats):
    player_name_dict_frequency = {}
    for player_scoring_stat in player_scoring_stats[1:]:
        player_scoring_stat_split = player_scoring_stat.split(',')
        player_first_name_score = player_scoring_stat_split[1].replace('"', '')
        player_last_name_score = player_scoring_stat_split[0].replace('"', '')
        if player_name_dict_frequency.get(player_first_name_score + ' ' + player_last_name_score) is not None:
            print player_first_name_score + ' ' + player_last_name_score
        else:
            player_name_dict_frequency.update({player_first_name_score + ' ' + player_last_name_score: 1})


if __name__ == '__main__':
    player_scoring_stats_dir = '../resource/ice_hockey_201819/NHL_player_1819_scoring.csv'
    player_scoring_stats = read_player_stats(player_scoring_stats_dir)
    # check_duplicate_name(player_scoring_stats)

    player_basic_info_dir = '../resource/ice_hockey_201819/player_info_2018_2019.json'
    with open(player_basic_info_dir, 'rb') as f:
        player_basic_info = json.load(f)
    match_player_name_id(player_basic_info, player_scoring_stats)
