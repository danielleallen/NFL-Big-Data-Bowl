import pandas as pd
import numpy as np
import math
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Load data files into pandas dataframes
df_track_2020 = pd.read_csv("data_sets/tracking2020.csv")
df_track_2019 = pd.read_csv("data_sets/tracking2019.csv")
df_track_2018 = pd.read_csv("data_sets/tracking2018.csv")

df_games = pd.read_csv("data_sets/games.csv")
df_plays = pd.read_csv("data_sets/plays.csv")


def process_play_data(df_plays, kickoff):
    '''
    Get the necessary columns from the plays dataframe and select only the rows that are for kickoffs where there
    was actually a return on the play
    Inputs:
        df_plays: pandas df of plays data
        kickoff: (boolean) True if we're selecting kickoff plays, False if we're selecting for punt plays
    Returns:
        kick_plays_df: Processed play dataframe
    '''
    if kickoff:       
        kick_plays_df = df_plays[(df_plays["specialTeamsPlayType"] == "Kickoff") & (df_plays["specialTeamsResult"] == "Return")]
    else:
        kick_plays_df = df_plays[(df_plays["specialTeamsPlayType"] == "Punt") & (df_plays["specialTeamsResult"] == "Return")]
    kick_plays_df = kick_plays_df[["gameId", "playId", "possessionTeam", "returnerId", "kickReturnYardage"]]  

    # Get rid of onside kicks
    kick_plays_df.dropna(axis=0, how='any', subset=['returnerId'], inplace=True)  
    # Get rid of kicks with multiple returners
    kick_plays_df.drop(kick_plays_df[kick_plays_df['returnerId'].str.contains(';')].index, inplace = True)  
    kick_plays_df["returnerId"] = kick_plays_df["returnerId"].astype('int')
    # Get rid of kicks with no return yards listed (likely because of fumble)
    kick_plays_df.dropna(axis=0, how='any', subset=['kickReturnYardage'], inplace=True) 
    
    return kick_plays_df


def process_tracking_data(df_track_2018, df_track_2019, df_track_2020, kickoff):
    '''
    Get the necessary data from the player tracking dataframes, standardize directions,
    and add additional features (velocity)
    Inputs: 
        df_track_2018, df_track_2019, df_track_2020: pandas df's of tracking data from correspinding seasons
        kickoff: (boolean) True if we're selecting kickoff plays, False if we're selecting for punt plays
    Returns:
        tracking_df: processed tracking dataframe that contains data from all three seasons
    '''
    tracking_df = pd.concat([df_track_2018, df_track_2019, df_track_2020])
    if kickoff:
        tracking_df = tracking_df[tracking_df["event"] == "kick_received"]
    else:
        tracking_df = tracking_df[tracking_df["event"] == "punt_received"]
    # Get rid of tracking data on football
    tracking_df.drop(tracking_df[tracking_df['team'] == "football"].index, inplace = True) 
    
    tracking_df['dir'] = np.mod(90 - tracking_df['dir'], 360)  # Change 0 degrees to be pointing downfield
    standardize_tracking_data(tracking_df)
    add_velocity_acc_vectors_tracking_data(tracking_df)
    
    tracking_df["nflId"] = tracking_df["nflId"].astype('int')
    tracking_df = tracking_df[['gameId','playId','nflId','team', 'x', 'y', 'v_x', 'v_y', 'a_x', 'a_y']]
        
    return tracking_df


def standardize_tracking_data(df_tracking):
    '''
    Helper for process_tracking_data(). Standardize the positions and directions of the play so that the returning 
    team is always going from left to right
    '''
    df_tracking.loc[df_tracking['playDirection'] == "right", 'x'] = 120-df_tracking.loc[df_tracking['playDirection'] == "right", 'x']
    df_tracking.loc[df_tracking['playDirection'] == "right", 'y'] = 160/3-df_tracking.loc[df_tracking['playDirection'] == "right", 'y']
    df_tracking.loc[df_tracking['playDirection'] == "right", 'dir'] = np.mod(180 + df_tracking.loc[df_tracking['playDirection'] == "right", 'dir'], 360)


def add_velocity_acc_vectors_tracking_data(df_tracking):
    '''
    Helper for process_tracking_data(). Use speed, acceleration, and direction to get the player's velocity 
    and acceleration in the x, y direction and add these as columns to df_tracking
    '''
    df_tracking["v_x"] = df_tracking["s"] * df_tracking["dir"].apply(math.radians).apply(math.cos)
    df_tracking["v_y"] = df_tracking["s"] * df_tracking["dir"].apply(math.radians).apply(math.sin)
    df_tracking["a_x"] = df_tracking["a"] * df_tracking["dir"].apply(math.radians).apply(math.cos)
    df_tracking["a_y"] = df_tracking["a"] * df_tracking["dir"].apply(math.radians).apply(math.sin)


def process_game_data(df_games):
    '''
    Select the necessary columns from the games df
    '''
    df_games_slim = df_games[['gameId', 'homeTeamAbbr', 'visitorTeamAbbr']]
    return df_games_slim


def merge_tables(df_games, df_kick_plays, df_tracking):
    '''
    Helper for process_data(). Merged the processed games, plays, and tracking data frames together
    '''
    game_play_merge = pd.merge(df_games, df_kick_plays, how='inner')
    all_merge = pd.merge(game_play_merge, df_tracking, how='inner')
    return all_merge


def add_player_side(play_track_df):
    '''
    Helper for process_data(). Add column saying whether player is on kicking_team, returning_team, or is the returner
    '''
    play_track_df["team_abbr"] = np.where(play_track_df["team"] == "home", play_track_df["homeTeamAbbr"], play_track_df["visitorTeamAbbr"])
    play_track_df["player_side"] = np.where(play_track_df["team_abbr"] == play_track_df["possessionTeam"], "kicking_team", "return_team")
    play_track_df["player_side"] = np.where(play_track_df["returnerId"] == play_track_df["nflId"], "returner", play_track_df["player_side"])


def process_data(df_track_2018, df_track_2019, df_track_2020, df_games, df_plays, kickoff=True):
    '''
    Process all data to create a single dataframe with all necessary information for the benchmark model
    '''
    kick_plays_df = process_play_data(df_plays, kickoff)
    df_games_slim = process_game_data(df_games)
    tracking_df = process_tracking_data(df_track_2018, df_track_2019, df_track_2020, kickoff)    
    play_track_df = merge_tables(df_games_slim, kick_plays_df, tracking_df)
    add_player_side(play_track_df)
    
    # Get rid of plays where there was not 22 players on field (Sillie billlies)
    grouped_df = play_track_df.groupby(["gameId", "playId"]) 
    play_track_df = grouped_df.filter(lambda x: x['nflId'].count() == 22)
    
    return play_track_df


if __name__ == '__main__':
    play_track_df = process_data(df_track_2018, df_track_2019, df_track_2020, df_games, df_plays)
    punt_play_track_df = process_data(df_track_2018, df_track_2019, df_track_2020, df_games, df_plays, kickoff= False)

    play_track_df.to_csv("play_track_kickoff.csv", index=False)
    punt_play_track_df.to_csv("play_track_punt.csv", index=False)

