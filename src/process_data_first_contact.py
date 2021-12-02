import pandas as pd
import numpy as np
import math
import pickle
from sklearn.model_selection import train_test_split


df_track_2020 = pd.read_csv("data_sets/tracking2020.csv")
df_track_2019 = pd.read_csv("data_sets/tracking2019.csv")
df_track_2018 = pd.read_csv("data_sets/tracking2018.csv")


df_games = pd.read_csv("data_sets/games.csv")
df_plays = pd.read_csv("data_sets/plays.csv")


kick_play_track_df = pd.read_csv("play_track_kickoff.csv")
punt_play_track_df = pd.read_csv("play_track_punt.csv")


def process_contact_tracking_data(df_track_2018, df_track_2019, df_track_2020):
    '''
    Process the tracking data to select for frames when there was first contact,
    and perform the necessary manipulations on the data
    '''
    tracking_df = pd.concat([df_track_2018, df_track_2019, df_track_2020])
    tracking_df = tracking_df[tracking_df["event"] == "first_contact"]
    # Get rid of tracking data on football
    tracking_df.drop(tracking_df[tracking_df['team'] == "football"].index, inplace = True) 
    
    tracking_df['dir'] = np.mod(90 - tracking_df['dir'], 360)  # Change 0 degrees to be pointing downfield
    standardize_tracking_data(tracking_df)
    add_velocity_vectors_tracking_data(tracking_df)
    
    tracking_df["nflId"] = tracking_df["nflId"].astype('int')
    tracking_df["contact_x"] = tracking_df["x"]
    tracking_df = tracking_df[['gameId','playId','nflId','contact_x']]
        
    return tracking_df


def standardize_tracking_data(df_tracking):
    '''
    Standardize the positions and directions of the play so that the returning team is always
    going from left to right
    '''
    df_tracking.loc[df_tracking['playDirection'] == "right", 'x'] = 120-df_tracking.loc[df_tracking['playDirection'] == "right", 'x']
    df_tracking.loc[df_tracking['playDirection'] == "right", 'y'] = 160/3-df_tracking.loc[df_tracking['playDirection'] == "right", 'y']
    df_tracking.loc[df_tracking['playDirection'] == "right", 'dir'] = np.mod(180 + df_tracking.loc[df_tracking['playDirection'] == "right", 'dir'], 360)


def add_velocity_vectors_tracking_data(df_tracking):
    '''
    Use speed and direction to get the player's velocity in the x, y direction and add these as columns to df
    '''
    df_tracking["v_x"] = df_tracking["s"] * df_tracking["dir"].apply(math.radians).apply(math.cos)
    df_tracking["v_y"] = df_tracking["s"] * df_tracking["dir"].apply(math.radians).apply(math.sin)


def process_first_contact_data(play_track_df, df_track_2018, df_track_2019, df_track_2020):
    '''
    Process all data to create a single dataframe with all necessary information for the 
    first contact model.
    '''
    contact_tracking_df = process_contact_tracking_data(df_track_2018, df_track_2019, df_track_2020)
    merged_catch_kick = pd.merge(play_track_df, contact_tracking_df, how='inner')
    
    merged_catch_kick['yards_to_contact'] = merged_catch_kick['contact_x'] - merged_catch_kick['x']
    
    # Get rid of plays where there was not 22 players on field (Sillie billlies)
    grouped_df = merged_catch_kick.groupby(["gameId", "playId"]) 
    merged_catch_kick = grouped_df.filter(lambda x: x['nflId'].count() == 22)  
    
    return merged_catch_kick
    

def create_X_tensor(play_track_df):
    '''
    Creates X input tensor for model. 
    Acknowledgement: This function as adapted from code written by Gordeev and Singer for their model.
                     https://www.kaggle.com/jccampos/nfl-2020-winner-solution-the-zoo
    Inputs:
        play_track_df: dataframe with all data needed for the model. Each row in the df
                       contains the tracking data for a single player on a given page
    Returns:    
        train_x: X input tensors for the model. Shape is (num_plays, 11, 10, 10)
    '''
    grouped_df = play_track_df.groupby(["gameId", "playId"])
    print(len(grouped_df))
    train_x = np.zeros([len(grouped_df),11,10,10])
    
    i = 0
    for name, group in grouped_df:
        [[returner_x, returner_y, returner_Vx, returner_Vy]] = group.loc[group.player_side=="returner",['x', 'y','v_x','v_y']].values

        kick_team_ids = group[group.player_side == "kicking_team"].index
        return_team_ids = group[group.player_side == "return_team"].index

        for j, kick_team_id in enumerate(kick_team_ids):
            [kick_team_x, kick_team_y, kick_team_Vx, kick_team_Vy] = group.loc[kick_team_id,['x', 'y','v_x','v_y']].values

            [kick_team_returner_x, kick_team_returner_y] = group.loc[kick_team_id,['x', 'y']].values - np.array([returner_x, returner_y])
            [kick_team_returner_Vx, kick_team_returner_Vy] = group.loc[kick_team_id,['v_x', 'v_y']].values - np.array([returner_Vx, returner_Vy])

            train_x[i,j,:,:4] = group.loc[return_team_ids,['v_x','v_y','x', 'y']].values - np.array([kick_team_x, kick_team_y, kick_team_Vx, kick_team_Vy])
            train_x[i,j,:,-6:] = [kick_team_returner_Vx, kick_team_returner_Vy, kick_team_returner_x, kick_team_returner_y, kick_team_Vx, kick_team_Vy]
        i += 1
    
    return train_x


# In[126]:


def create_y_train(play_track_df, min_idx_y, max_idx_y):
    '''
    Create dataframe consisting of the yards to first contact on each play. Considering that the model
    returns an array of probabilities where each index in the array corresponds to yards to contact,
    the dataframe also contains a columns with the corresponding array index for 
    yards to first contact.  
    Acknowledgement: This function as adapted from code written by Gordeev and Singer for their model.
                     https://www.kaggle.com/jccampos/nfl-2020-winner-solution-the-zoo
    Iputs: 
        play_track_df: dataframe with all data needed for the model. Each row in the df
                       contains the tracking data for a single player on a given page

        min_idx_y: Minimum index we allow for yards gained
        max_idx_y: Maximum index we allow for yards gained
    Returns: 
        train_y: df with with columns for yards to first contact and the yards to contact's inedx for each play
    '''  
    train_y = play_track_df[play_track_df['player_side'] == 'returner'][["gameId", "playId", "yards_to_contact"]]
    
    train_y['YardIndex'] = train_y["yards_to_contact"].apply(lambda val: val + 99)
    train_y['YardIndexClipped'] = train_y['YardIndex'].apply(
        lambda val: min_idx_y if val < min_idx_y else max_idx_y if val > max_idx_y else val)
    
    return train_y


def create_y_tensor(train_y, min_idx_y, max_idx_y):
    '''
    For each play, create a one-hot encoded vector for yards to first contact. This will act 
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


def get_input_data(play_track_df, output_tag, min_idx_y, max_idx_y, test_size=0.1, save_tensors=True):
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
    X_tensor = create_X_tensor(play_track_df)
    print("X Shape", X_tensor.shape)
    
    train_y = create_y_train(play_track_df, min_idx_y, max_idx_y)
    y_tensor = create_y_tensor(train_y, min_idx_y, max_idx_y)
    print("y Shape", y_tensor.shape)
    
    output_tag += str(min_idx_y) + "_" + str(max_idx_y)
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = train_test_split(X_tensor, y_tensor, test_size=test_size)
    
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


def get_input_data_combo(kick_play_track_df, punt_play_track_df, output_tag, min_idx_y, max_idx_y, test_size=0.1):
    '''
    Create the X and y train and test tensors for when inpit shold be a combination of kickoffs or punts
    Inputs:
        kick_play_track_df: dataframe with all data kickoff data needed for the model. Each row in the df
                       contains the tracking data for a single player on a given page
        punt_play_track_df: dataframe with all data kickoff data needed for the model. Each row in the df
                       contains the tracking data for a single player on a given page
        output_tag: labels to save tensors with. Should contain "comb", "kick", or "punt"
        min_idx_y: (int) Minimum index we allow for yards gained
        max_idx_y: (int) Maximum index we allow for yards gained
        test_size: (float) Proportion of data to use as test set
    Returns: 
        X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor: Input tensors for model
    '''
    X_train_kick_tensor, X_test_kick_tensor, y_train_kick_tensor, y_test_kick_tensor = get_input_data(kick_play_track_df, "kick", min_idx_y, max_idx_y, test_size=test_size, save_tensors=False)
    X_train_punt_tensor, X_test_punt_tensor, y_train_punt_tensor, y_test_punt_tensor = get_input_data(punt_play_track_df, "punt", min_idx_y, max_idx_y, test_size=test_size, save_tensors=False)
    
    X_train_comb_tensor = np.concatenate((X_train_kick_tensor, X_train_punt_tensor))
    X_test_comb_tensor = np.concatenate((X_test_kick_tensor, X_test_punt_tensor))
    y_train_comb_tensor = np.concatenate((y_train_kick_tensor, y_train_punt_tensor))
    y_test_comb_tensor = np.concatenate((y_test_kick_tensor, y_test_punt_tensor))
    
    print("X_train_combo", X_train_comb_tensor.shape)
    print("X_test_combo", X_test_comb_tensor.shape)
    print("y_train_combo", y_train_comb_tensor.shape)
    print("y_test_combo", y_test_comb_tensor.shape)
    
    output_tag += str(min_idx_y) + "_" + str(max_idx_y)
    with open(f'input_tensors/X_tensor_comb_{output_tag}_train.data', 'wb') as f:
        pickle.dump(X_train_comb_tensor, f)
    with open(f'input_tensors/X_tensor_comb_{output_tag}_test.data', 'wb') as f:
        pickle.dump(X_test_comb_tensor, f)
    with open(f'input_tensors/y_tensor_comb_{output_tag}_train.data', 'wb') as f:
        pickle.dump(y_train_comb_tensor, f)
    with open(f'input_tensors/y_tensor_comb_{output_tag}_test.data', 'wb') as f:
        pickle.dump(y_test_comb_tensor, f)  
    
    return X_train_comb_tensor, X_test_comb_tensor, y_train_comb_tensor, y_test_comb_tensor
    
    
if __name__ == '__main__':  
    kick_contact_df = process_first_contact_data(kick_play_track_df, df_track_2018, df_track_2019, df_track_2020)
    punt_contact_df = process_first_contact_data(punt_play_track_df, df_track_2018, df_track_2019, df_track_2020)

    get_input_data(kick_contact_df, "contact_kick", 94, 154, test_size=0.1, save_tensors=True)
    get_input_data(punt_contact_df, "contact_punt", 94, 154, test_size=0.1, save_tensors=True)
    get_input_data_combo(kick_contact_df, punt_contact_df, "comb_contact", 94, 154, test_size=0.1)



