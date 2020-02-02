import datetime
import tensorflow as tf
import os
import scipy.io as sio
import numpy as np
import copy
import random
from array import array

import csv 

import sys
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + "/..")

# def make_one_hot_action(event_9_to_1):
#     event_9_to_1_with_one_hot_action = ''

#     for i in range(len(event_9_to_1)):
#         if i >= 16 and i <= 58:
#             if float(event_9_to_1[i]) > 0:
#                 event_9_to_1_with_one_hot_action = event_9_to_1_with_one_hot_action + '1,'
#             else:
#                 event_9_to_1_with_one_hot_action = event_9_to_1_with_one_hot_action + '0,'



def make_csv_str(tokens):
    csv_str = ''

    for token in tokens:
        # do not need to denormalize, since printing tree will do 
        csv_str = csv_str + str(token).strip() + ','

    return csv_str


def read_csv(csvfile, Q_file_Writer, impact_file_Writer, standard_shot_index):
    with open(csvfile, 'r') as fp:
        csv_file_iterator = csv.reader(fp, delimiter=',')

        row_number = 0

        for row in csv_file_iterator:
            row_number = row_number + 1
            row_is_blank = True

            impact = row[0]
            Q = row[1]
            
            # read history events of the row
            event_count = 10
            while event_count > 0:
                event_count = event_count - 1

                # first 2 are impact and Q 
                # 16: state features 
                # 43: actions
                # 2: home and away
                front_index = 2 + (9 - event_count) * (16 + 43 + 2)
                end_index = 63 + (9 - event_count) * (16 + 43 + 2)

                current_event = row[front_index:end_index]

                # print(current_event)

                # fix home and away and move them into state features
                home = current_event[-2]
                away = current_event[-1]

                state_features = current_event[0:16]
                state_features.append(home)
                state_features.append(away)

                actions = current_event[16:-2]

                # print(state_features)
                # print(actions)

                # for headers
                if row_number == 1:
                    # write Q and impact once 
                    if event_count == 9:
                        impact_file_Writer.write(impact + ',')
                        Q_file_Writer.write(Q + ',')
                        row_is_blank = False

                    # no action for event0
                    if event_count == 0:
                        # use [:-1] because we do not want the last comma 
                        impact_file_Writer.write(make_csv_str(state_features)[:-1])
                        Q_file_Writer.write(make_csv_str(state_features)[:-1])
                    else:
                        impact_file_Writer.write(make_csv_str(state_features) + make_csv_str(actions))
                        Q_file_Writer.write(make_csv_str(state_features) + make_csv_str(actions))

                # for numbers
                else:
                    # for rows with standard_shot0 is true
                    if float(row[standard_shot_index]) == 1:
                        # print('standard_shot: ', row[standard_shot_index])

                        # write Q and impact once 
                        if event_count == 9:
                            impact_file_Writer.write(impact + ',')
                            Q_file_Writer.write(Q + ',')
                            row_is_blank = False

                        # no action for event0
                        if event_count == 0:
                            # use [:-1] because we do not want the last comma 
                            impact_file_Writer.write(make_csv_str(state_features)[:-1])
                            Q_file_Writer.write(make_csv_str(state_features)[:-1])
                        else:
                            impact_file_Writer.write(make_csv_str(state_features) + make_csv_str(actions))
                            Q_file_Writer.write(make_csv_str(state_features) + make_csv_str(actions))
            
            if row_is_blank == False:
                Q_file_Writer.write('\n')
                impact_file_Writer.write('\n')
        
        fp.close()

if __name__ == '__main__':
    # number of state features: 18 = 16 + 2, last 2 are home and away
    # number of actions: 43
    # in total: 2 + (18 + 43) * 10 = impact + Q + 610
    
    Q_data_DIR = '/Users/xiangyusun/Development/LMUT/csv_files/soccer/'

    file_name = Q_data_DIR + '/csv_from_galen/shot_impact_Q_states_features_history_soccer.csv'
    
    Q_file_name = 'Q_standard_shot_soccer.csv'
    impact_file_name = 'impact_standard_shot_soccer.csv'

    Q_file_Writer = open(Q_data_DIR + '/' + Q_file_name, 'w')
    impact_file_Writer = open(Q_data_DIR + '/' + impact_file_name, 'w')

    # action standard_shot0 index = -20 + (-2), last 2 are home and away
    standard_shot_index = -22

    read_csv(file_name, Q_file_Writer, impact_file_Writer, standard_shot_index)

    Q_file_Writer.close()
    impact_file_Writer.close()

    print('Done')

    

