import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split


kick_play_track_df = pd.read_csv("play_track_kickoff.csv")
punt_play_track_df = pd.read_csv("play_track_punt.csv")
players_df = pd.read_csv("data_sets/players.csv")[["nflId", "weight"]]


def add_momentum_force(play_track_df, players_df):
    '''
    Add a columnw for momentum and force to the processed dataframe that contained the data
    needed for the original benchmark model
    Inputs:
        play_track_df: dataframe with all data needed for the benchmark model. Each row in the df
                       contains the tracking data for a single player on a given play
        players_df: dataframe containing player related information. We are interested
                    in weight
    Returns:
        play_track_momentum_df: dataframe with all data needed for the force momentum model. Each row in the df
                                contains the tracking data for a single player on a given play
    '''
    lb_to_kg = 0.453592
    play_track_pf_df = pd.merge(play_track_df, players_df, how='inner')   
    play_track_pf_df["p_x"] = play_track_pf_df["v_x"] * lb_to_kg * play_track_pf_df["weight"]
    play_track_pf_df["p_y"] = play_track_pf_df["v_y"] * lb_to_kg * play_track_pf_df["weight"]
    play_track_pf_df["f_x"] = play_track_pf_df["a_x"] * lb_to_kg * play_track_pf_df["weight"]
    play_track_pf_df["f_y"] = play_track_pf_df["a_y"] * lb_to_kg * play_track_pf_df["weight"]
    
    return play_track_pf_df


def create_X_tensor_momentum_force(play_track_df):
    '''
    Creates X input tensor for force momentum model. 
    Acknowledgement: This function as adapted from code written by Gordeev and Singer for their model.
                     https://www.kaggle.com/jccampos/nfl-2020-winner-solution-the-zoo
    Inputs:
        play_track_df: dataframe with all data needed for the model. Each row in the df
                       contains the tracking data for a single player on a given page
    Returns:    
        train_x: X input tensors for the model. Shape is (num_plays, 11, 10, 14)
    '''
    grouped_df = play_track_df.groupby(["gameId", "playId"])
    print(len(grouped_df))
    train_x = np.zeros([len(grouped_df),11,10,14])
    
    i = 0
    for name, group in grouped_df:
        [[returner_x, returner_y, returner_Px, returner_Py, returner_Fx, returner_Fy]] = group.loc[group.player_side=="returner",['x', 'y','p_x','p_y','f_x','f_y']].values

        kick_team_ids = group[group.player_side == "kicking_team"].index
        return_team_ids = group[group.player_side == "return_team"].index

        for j, kick_team_id in enumerate(kick_team_ids):
            [kick_team_x, kick_team_y, kick_team_Px, kick_team_Py, kick_team_Fx, kick_team_Fy] = group.loc[kick_team_id,['x', 'y','p_x','p_y','f_x','f_y']].values

            [kick_team_returner_x, kick_team_returner_y] = group.loc[kick_team_id,['x', 'y']].values - np.array([returner_x, returner_y])
            [kick_team_returner_Px, kick_team_returner_Py] = group.loc[kick_team_id,['p_x', 'p_y']].values - np.array([returner_Px, returner_Py])
            [kick_team_returner_Fx, kick_team_returner_Fy] = group.loc[kick_team_id,['f_x', 'f_y']].values - np.array([returner_Fx, returner_Fy])
            
            train_x[i,j,:,:6] = group.loc[return_team_ids,['p_x','p_y','x', 'y', 'f_x', 'f_y']].values - np.array([kick_team_x, kick_team_y, kick_team_Px, kick_team_Py, kick_team_Fx, kick_team_Fy])
            train_x[i,j,:,-8:] = [kick_team_returner_Px, kick_team_returner_Py, kick_team_returner_Fx, kick_team_returner_Fy, kick_team_returner_x, kick_team_returner_y, kick_team_Px, kick_team_Py]
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


def get_input_data_momentum_force(play_track_df, output_tag, min_idx_y, max_idx_y, test_size=0.1, save_tensors=True):
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
    X_tensor = create_X_tensor_momentum_force(play_track_df)
    print("X Shape", X_tensor.shape)
    
    train_y = create_y_train(play_track_df, min_idx_y, max_idx_y)
    y_tensor = create_y_tensor(train_y, min_idx_y, max_idx_y)
    print("y Shape", y_tensor.shape)
    
    output_tag += "_" + str(min_idx_y) + "_" + str(max_idx_y)
     
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


def get_input_data_momentum_force_combo(kick_play_track_df, punt_play_track_df, output_tag, min_idx_y, max_idx_y, test_size=0.1):
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
    X_train_kick_tensor, X_test_kick_tensor, y_train_kick_tensor, y_test_kick_tensor = get_input_data_momentum_force(kick_play_track_df, "kick", min_idx_y, max_idx_y, test_size=test_size, save_tensors=False)
    X_train_punt_tensor, X_test_punt_tensor, y_train_punt_tensor, y_test_punt_tensor = get_input_data_momentum_force(punt_play_track_df, "punt", min_idx_y, max_idx_y, test_size=test_size, save_tensors=False)
    
    X_train_comb_tensor = np.concatenate((X_train_kick_tensor, X_train_punt_tensor))
    X_test_comb_tensor = np.concatenate((X_test_kick_tensor, X_test_punt_tensor))
    y_train_comb_tensor = np.concatenate((y_train_kick_tensor, y_train_punt_tensor))
    y_test_comb_tensor = np.concatenate((y_test_kick_tensor, y_test_punt_tensor))
    
    print("X_train_combo", X_train_comb_tensor.shape)
    print("X_test_combo", X_test_comb_tensor.shape)
    print("y_train_combo", y_train_comb_tensor.shape)
    print("y_test_combo", y_test_comb_tensor.shape)
    
    output_tag += "_" + str(min_idx_y) + "_" + str(max_idx_y)
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
    kick_play_track_pf_df = add_momentum_force(kick_play_track_df, players_df)
    punt_play_track_pf_df = add_momentum_force(punt_play_track_df, players_df)

    get_input_data_momentum_force_combo(kick_play_track_pf_df, punt_play_track_pf_df, "comb_force_momentum", 94, 154)
    get_input_data_momentum_force(kick_play_track_pf_df, "kick_force_momentum", 94, 154, test_size=0.1, save_tensors=True)
    get_input_data_momentum_force(punt_play_track_pf_df, "punt_force_momentum", 94, 154, test_size=0.1, save_tensors=True)

