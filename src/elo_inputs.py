import pandas as pd
import numpy as np
import pickle
import math
from sklearn.model_selection import train_test_split

kick_play_track_df = pd.read_csv("play_track_kickoff.csv")
punt_play_track_df = pd.read_csv("play_track_punt.csv")
elo_df = pd.read_csv("elo.csv")


def calc_probability(rating1, rating2):
    '''
    Calculates the probability of player 1 winning a matchup against player 2
    based on their ratings
    '''
    prob = 1.0 * 1.0 / (1 + 1.0 * math.pow(10, 1.0 * (rating2 - rating1) / 400))
    return prob


def weighted_probs(returner_rating, tackler_rating, prob_tackle):
    '''
    Calculates the probabilities of the returner and tackler winning the matchup factoring in the general
    probability that any given tackle is made 
    Inputs:
        returner_rating: (float) returner's current rating
        tackler_rating: (float) tackler's current rating
        prob_tackle: (float) general probability that any given tackle is made
    Returns:
        adjusted_returner, adjusted_tackler: 
    '''
    returner_prob = calc_probability(returner_rating, tackler_rating)
    tackler_prob = calc_probability(tackler_rating, returner_rating)
    
    returner_prob_weighted = returner_prob * (1-prob_tackle)
    tackler_prob_weighted = tackler_prob * (prob_tackle)
    
    adjusted_returner = returner_prob_weighted / (returner_prob_weighted + tackler_prob_weighted)
    adjusted_tackler = tackler_prob_weighted / (returner_prob_weighted + tackler_prob_weighted)

    return adjusted_returner, adjusted_tackler


def create_X_tensor_elo(play_track_df, prob_tackle):
    '''
    Creates X input tensor for elo model. 
    Acknowledgement: This function as adapted from code written by Gordeev and Singer for their model.
                     https://www.kaggle.com/jccampos/nfl-2020-winner-solution-the-zoo
    Inputs:
        play_track_df: dataframe with all data needed for the model. Each row in the df
                       contains the tracking data for a single player on a given page
        prob_tackle: (float) general probability that any given tackle is made
    Returns:    
        train_x: X input tensors for the model. Shape is (num_plays, 11, 10, 11)
    '''
    grouped_df = play_track_df.groupby(["gameId", "playId"])
    print(len(grouped_df))
    train_x = np.zeros([len(grouped_df),11,10,11])
    
    i = 0
    for name, group in grouped_df:
        [[returner_x, returner_y, returner_Vx, returner_Vy, returner_elo]] = group.loc[group.player_side=="returner",['x', 'y','v_x','v_y', 'returner_elo']].values

        kick_team_ids = group[group.player_side == "kicking_team"].index
        return_team_ids = group[group.player_side == "return_team"].index
    
        for j, kick_team_id in enumerate(kick_team_ids):
            [kick_team_x, kick_team_y, kick_team_Vx, kick_team_Vy, kick_team_elo] = group.loc[kick_team_id,['x', 'y','v_x','v_y', 'tackler_elo']].values

            [kick_team_returner_x, kick_team_returner_y] = group.loc[kick_team_id,['x', 'y']].values - np.array([returner_x, returner_y])
            [kick_team_returner_Vx, kick_team_returner_Vy] = group.loc[kick_team_id,['v_x', 'v_y']].values - np.array([returner_Vx, returner_Vy])
            elo_prob_tackle = weighted_probs(returner_elo, kick_team_elo, prob_tackle)[1]

            train_x[i,j,:,:4] = group.loc[return_team_ids,['v_x','v_y','x', 'y']].values - np.array([kick_team_x, kick_team_y, kick_team_Vx, kick_team_Vy])
            train_x[i,j,:,-7:] = [kick_team_returner_Vx, kick_team_returner_Vy, kick_team_returner_x, kick_team_returner_y, kick_team_Vx, kick_team_Vy, elo_prob_tackle]
        i += 1
    
    return train_x


def create_y_train(play_track_df, min_idx_y, max_idx_y):
    '''
    Create dataframe consisting of the yards gained on each play. Considering that the model
    returns an array of probabilities where each index in the array corresponds to yards gained,
    the dataframe also contains a columns with the corresponding array index for 
    yards gained.  
    Acknowledgement: This function as adapted from code written by Gordeev and Singer for their model.
                     https://www.kaggle.com/jccampos/nfl-2020-winner-solution-the-zoo
    Iputs: 
        play_track_df: dataframe with all data needed for the model. Each row in the df
                       contains the tracking data for a single player on a given page

        min_idx_y: Minimum index we allow for yards gained
        max_idx_y: Maximum index we allow for yards gained
    Returns: 
        train_y: df with with columns for yards gained and the yard's gained inedx for each play
    ''' 
    train_y = play_track_df.groupby(["gameId", "playId"])["kickReturnYardage"].mean()
    train_y = train_y.to_frame()
    train_y.reset_index(level=["gameId", "playId"], inplace=True)
    
    train_y['YardIndex'] = train_y["kickReturnYardage"].apply(lambda val: val + 99)
    train_y['YardIndexClipped'] = train_y['YardIndex'].apply(
        lambda val: min_idx_y if val < min_idx_y else max_idx_y if val > max_idx_y else val)
    
    return train_y


def create_y_tensor(train_y, min_idx_y, max_idx_y):
    '''
    For each play, create a one-hot encoded vector for yards gained. This will act 
    as y input to the model
    Acknowledgement: This function as adapted from code written by Gordeev and Singer for their model.
                     https://www.kaggle.com/jccampos/nfl-2020-winner-solution-the-zoo
    Inputs:
        play_track_df: dataframe with all data needed for the model. Each row in the df
                       contains the tracking data for a single player on a given page

        min_idx_y: Minimum index we allow for yards gained
        max_idx_y: Maximum index we allow for yards gained
    Returns: numpy array of shape (num_plays, num_y_classes) where num_y_classes
             is the number of different yards that can be predicted
    '''
    num_classes_y = max_idx_y - min_idx_y + 1
    y_vals = train_y["YardIndexClipped"].values
    y_tensor = np.zeros((len(y_vals), num_classes_y), np.int32)
    for i, yards in enumerate(y_vals):
        y_tensor[(i, yards.astype(np.int32) - min_idx_y)] = 1
    
    return y_tensor


def get_input_data_elo(play_track_df, output_tag, min_idx_y, max_idx_y, prob_tackle, test_size=0.1, save_tensors=True):
    '''
    Create the X and y train and test tensors for either kickoffs or punts
    Inputs:
        play_track_df: dataframe with all data needed for the model. Each row in the df
                       contains the tracking data for a single player on a given page
        output_tag: labels to save tensors with. Should contain "comb", "kick", or "punt"
        min_idx_y: (int) Minimum index we allow for yards gained
        max_idx_y: (int) Maximum index we allow for yards gained
        test_size: (float) Proportion of data to use as test set
        save_tensors: (Boolean) True if you want to save tensors to file.
    Returns: 
        X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor: Input tensors for model
    '''
    X_tensor = create_X_tensor_elo(play_track_df, prob_tackle)
    print("X Shape", X_tensor.shape)
    
    train_y = create_y_train(play_track_df, min_idx_y, max_idx_y)
    y_tensor = create_y_tensor(train_y, min_idx_y, max_idx_y)
    print("y Shape", y_tensor.shape)
    
    output_tag += str(min_idx_y) + "_" + str(max_idx_y)   
    
    # Load indices of train test split based if on the data is for kickoffs or punts 
    if "kick" in output_tag:
        data_desc = "kick"
    else:
        data_desc = "punt"       
    
    with open(f'train_test_idxs/X_{data_desc}_train_idxs.data', 'rb') as f:
        X_train_idx = pickle.load(f)
    with open(f'train_test_idxs/X_{data_desc}_test_idxs.data', 'rb') as f:
        X_test_idx = pickle.load(f)
    with open(f'train_test_idxs/y_{data_desc}_train_idxs.data', 'rb') as f:
        y_train_idx = pickle.load(f)
    with open(f'train_test_idxs/y_{data_desc}_test_idxs.data', 'rb') as f:
        y_test_idx = pickle.load(f)
    
    X_train_tensor = X_tensor[X_train_idx]
    X_test_tensor = X_tensor[X_test_idx]
    y_train_tensor = y_tensor[y_train_idx]
    y_test_tensor = y_tensor[y_test_idx]
    
    print("Train X", X_train_tensor.shape)
    print("Test X", X_test_tensor.shape)
    print("Train y", y_train_tensor.shape)
    print("Test y", y_test_tensor.shape)
    
    if save_tensors:
        with open(f'input_tensors/X_tensor_{output_tag}_train.data', 'wb') as f:
            pickle.dump(X_train_tensor, f)
        with open(f'input_tensors/X_tensor_{output_tag}_test.data', 'wb') as f:
            pickle.dump(X_test_tensor, f)
        with open(f'input_tensors/y_tensor_{output_tag}_train.data', 'wb') as f:
            pickle.dump(y_train_tensor, f)
        with open(f'input_tensors/y_tensor_{output_tag}_test.data', 'wb') as f:
            pickle.dump(y_test_tensor, f)
    
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor


def get_input_data_elo_combo(kick_play_track_df, punt_play_track_df, output_tag, min_idx_y, max_idx_y, prob_tackle, test_size=0.1):
    '''
    Create the X and y train and test tensors for either kickoffs or punts
    Inputs:
        play_track_df: dataframe with all data needed for the model. Each row in the df
                       contains the tracking data for a single player on a given page
        output_tag: labels to save tensors with. Should contain "comb", "kick", or "punt"
        min_idx_y: (int) Minimum index we allow for yards gained
        max_idx_y: (int) Maximum index we allow for yards gained
        test_size: (float) Proportion of data to use as test set
        save_tensors: (Boolean) True if you want to save tensors to file.
    Returns: 
        X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor: Input tensors for model
    '''
    X_train_kick_tensor, X_test_kick_tensor, y_train_kick_tensor, y_test_kick_tensor = get_input_data_elo(kick_play_track_df, "kick", min_idx_y, max_idx_y, prob_tackle, test_size=test_size, save_tensors=False)
    X_train_punt_tensor, X_test_punt_tensor, y_train_punt_tensor, y_test_punt_tensor = get_input_data_elo(punt_play_track_df, "punt", min_idx_y, max_idx_y, prob_tackle, test_size=test_size, save_tensors=False)
    
    X_train_comb_tensor = np.concatenate((X_train_kick_tensor, X_train_punt_tensor))
    X_test_comb_tensor = np.concatenate((X_test_kick_tensor, X_test_punt_tensor))
    y_train_comb_tensor = np.concatenate((y_train_kick_tensor, y_train_punt_tensor))
    y_test_comb_tensor = np.concatenate((y_test_kick_tensor, y_test_punt_tensor))
    
    print("X_train_combo", X_train_comb_tensor.shape)
    print("X_test_combo", X_test_comb_tensor.shape)
    print("y_train_combo", y_train_comb_tensor.shape)
    print("y_test_combo", y_test_comb_tensor.shape)
    
    output_tag += str(min_idx_y) + "_" + str(max_idx_y)
    with open(f'input_tensors/X_tensor_{output_tag}_train.data', 'wb') as f:
        pickle.dump(X_train_comb_tensor, f)
    with open(f'input_tensors/X_tensor_{output_tag}_test.data', 'wb') as f:
        pickle.dump(X_test_comb_tensor, f)
    with open(f'input_tensors/y_tensor_{output_tag}_train.data', 'wb') as f:
        pickle.dump(y_train_comb_tensor, f)
    with open(f'input_tensors/y_tensor_{output_tag}_test.data', 'wb') as f:
        pickle.dump(y_test_comb_tensor, f)   
    
    return X_train_comb_tensor, X_test_comb_tensor, y_train_comb_tensor, y_test_comb_tensor


if __name__ == '__main__':
    kick_play_track_elo = pd.merge(kick_play_track_df, elo_df, how='left')
    punt_play_track_elo = pd.merge(punt_play_track_df, elo_df, how='left')

    # Replace all missing Elo values with 1000 since that is the rating all players start with
    kick_play_track_elo.fillna(value={"returner_elo": 1000, "tackler_elo": 1000}, inplace=True)
    punt_play_track_elo.fillna(value={"returner_elo": 1000, "tackler_elo": 1000}, inplace=True)

    print(kick_play_track_elo)
    print(punt_play_track_elo)
    
    get_input_data_elo_combo(kick_play_track_elo, punt_play_track_elo, "comb_elo", 94, 154, 0.7321)
    get_input_data_elo(kick_play_track_elo, "kick_elo", 94, 154, 0.7321, test_size=0.1, save_tensors=True)
    get_input_data_elo(punt_play_track_elo, "punt_elo", 94, 154, 0.7321, test_size=0.1, save_tensors=True)




