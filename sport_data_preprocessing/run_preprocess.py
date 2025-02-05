import json

from preprocess import Preprocess
from build_seq import process_seq_all

if __name__ == '__main__':
    hockey_data_dir = '/cs/oschulte/2019-icehockey-data/2018-2019/'
    # hockey_data_dir = '/Users/xiangyusun/Desktop/2019-icehockey-data-small/2018-2019/'

    save_data_dir = '/cs/oschulte/xiangyus/2019-icehockey-data-preprocessed/2018-2019/'
    # save_data_dir = '/Users/xiangyusun/Desktop/2019-icehockey-data-preprocessed/2018-2019/'

    player_basic_info_dir = '../resource/ice_hockey_201819/player_info_2018_2019.json'
    # player_basic_info_dir = '/Users/xiangyusun/Development/sport-analytic-variational-embedding/resource/ice_hockey_201819/player_info_2018_2019.json'

    team_info_dir = '../resource/ice_hockey_201819/teams_NHL.json'
    # team_info_dir = '/Users/xiangyusun/Development/sport-analytic-variational-embedding/resource/ice_hockey_201819/teams_NHL.json'

    game_date_dir = '../resource/ice_hockey_201819/game_dates_2018_2019.json'
    # game_date_dir = '/Users/xiangyusun/Development/sport-analytic-variational-embedding/resource/ice_hockey_201819/game_dates_2018_2019.json'

    with open(player_basic_info_dir, 'rb') as f:
        player_basic_info_dict = json.load(f)

    with open(team_info_dir, 'rb') as f:
        team_info_dict = json.load(f)

    with open(game_date_dir, 'rb') as f:
        game_date_dict = json.load(f)

    prep = Preprocess(hockey_data_dir=hockey_data_dir, save_data_dir=save_data_dir,
                      player_basic_info_dict=player_basic_info_dict, team_info_dict=team_info_dict,
                      game_date_dict=game_date_dict)
    scaler = prep.scale_allgame_features()
    wrong_files = prep.process_all(scaler=scaler)
    prep.complement_data()
    process_seq_all(save_data_dir=save_data_dir)
    
    print 'wrong files skipped are {0}'.format(str(wrong_files))

    # prep.generate_player_information(store_player_info_dir='../resource/ice_hockey_201819/player_info_2018_2019.json')
