MODEL_TYPE = "v4"
MAX_TRACE_LENGTH = 2
# FEATURE_NUMBER = 25
FEATURE_NUMBER = 39 # 12 + 27
BATCH_SIZE = 32
GAMMA = 1
H_SIZE = 512
USE_HIDDEN_STATE = False
model_train_continue = True
SCALE = True
FEATURE_TYPE = 5
ITERATE_NUM = 30
learning_rate = 1e-4
SPORT = "NHL"
directory_generated_Q_data = "Q_data"
# save_mother_dir = "/Users/xiangyusun/Desktop"
save_mother_dir = "/cs/oschulte/xiangyus/DRL-ice-hockey-saves"
action_all = ['assist',
              'block',
              'carry',
              'check',
              'controlledbreakout',
              'controlledentryagainst',
              'dumpin',
              'dumpinagainst',
              'dumpout',
              'faceoff',
              'goal',
              'icing',
              'lpr',
              'offside',
              'pass',
              'pass1timer',
              'penalty',
              'pressure',
              'pscarry',
              'pscheck',
              'pslpr',
              'pspuckprotection',
              'puckprotection',
              'reception',
              'receptionprevention',
              'shot',
              'shot1timer']
