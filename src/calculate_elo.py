import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import pickle


df_plays = pd.read_csv("data_sets/plays.csv")
df_pff = pd.read_csv("data_sets/PFFScoutingData.csv")
df_games = pd.read_csv("data_sets/games.csv")
df_players = pd.read_csv("data_sets/players.csv")

df_track_2020 = pd.read_csv("data_sets/tracking2020.csv")
df_track_2019 = pd.read_csv("data_sets/tracking2019.csv")
df_track_2018 = pd.read_csv("data_sets/tracking2018.csv")


def process_play_data(df_plays):
    '''
    Get the necessary columns from the plays dataframe and select only the rows that are for kickoffs where there
    was actually a return on the play
    '''  
    kick_plays_df = df_plays[((df_plays["specialTeamsPlayType"] == "Kickoff") | (df_plays["specialTeamsPlayType"] == "Punt")) & (df_plays["specialTeamsResult"] == "Return")]
    kick_plays_df = kick_plays_df[["gameId", "playId", "specialTeamsPlayType","possessionTeam", "returnerId", "kickReturnYardage"]]  
    # Get rid of onside kicks
    kick_plays_df.dropna(axis=0, how='any', subset=['returnerId'], inplace=True)  
    # Get rid of kicks with multiple returners
    kick_plays_df.drop(kick_plays_df[kick_plays_df['returnerId'].str.contains(';')].index, inplace = True)  
    kick_plays_df["returnerId"] = kick_plays_df["returnerId"].astype('int')
    # Get rid of kicks with no return yards listed (likely because of fumble)
    kick_plays_df.dropna(axis=0, how='any', subset=['kickReturnYardage'], inplace=True) 
    
    return kick_plays_df


def process_pff_data(df_pff):
    '''
    Select the necessary columns from df_pff
    '''
    pff_tackle_df = df_pff[["gameId", "playId", "missedTackler", "tackler"]]
    return pff_tackle_df


def create_jersey_map(df_tracking, df_games):
    '''
    Create df that maps game and jersey number from that game to the correct nflId (player's id)
    Acknowledement: The code from this function was taken directly from the following discussion post
                    https://www.kaggle.com/dhritiyandapally/speed-on-kickoff-plays-across-surfaces-python
    Inputs:
        df_tracking: Contains tracking data from all three seasons
        df_games: contains game information
    Returns:
        df_jerseyMap: df that maps game and jersey number from that game to the correct nflId (player's id)
    '''
    #selecting variables of interest & dropping duplicates - jersey # is constant throughout game
    df_jerseyMap = df_tracking.drop_duplicates(subset = ["gameId", "team", "jerseyNumber", "nflId"])
    #joining to games
    df_jerseyMap = pd.merge(df_jerseyMap, df_games, left_on=['gameId'], right_on =['gameId'])
    #getting name of team
    conditions = [
        (df_jerseyMap['team'] == "home"),
        (df_jerseyMap['team'] != "home"),
    ]
    values = [df_jerseyMap['homeTeamAbbr'], df_jerseyMap['visitorTeamAbbr']]
    #adjusting jersey number so that it includes 0 when < 10
    df_jerseyMap['team'] = np.select(conditions, values)
    df_jerseyMap['jerseyNumber'] = df_jerseyMap['jerseyNumber'].astype(str)
    df_jerseyMap.loc[df_jerseyMap['jerseyNumber'].map(len) < 4, 'jerseyNumber'] = "0"+df_jerseyMap.loc[df_jerseyMap['jerseyNumber'].map(len) < 4, 'jerseyNumber'].str[:2]
    df_jerseyMap['jerseyNumber'] = df_jerseyMap['jerseyNumber'].str[:2]
    #getting team and jersey
    df_jerseyMap['teamJersey'] = df_jerseyMap['team'] + ' ' + df_jerseyMap['jerseyNumber'].str[:2]
    #map to merge nflId to teamJersey
    df_jerseyMap = df_jerseyMap[['gameId', 'nflId', 'teamJersey']]
    df_jerseyMap = df_jerseyMap.sort_values(['gameId', 'nflId', 'teamJersey'])

    return df_jerseyMap


def get_matchups(merged_plays_pff, df_jerseyMap):
    '''
    Finds all matchups between tacklers and returners and stores them in dataframe
    along with the results of matchup
    Inputs:
        merged_plays_pff: df with tackle information per play
        df_jerseyMap: df that maps game and jersey number from that game to the correct nflId (player's id)
    Returns:
        matchup_df: df of all the matchup. Each row contains the tackler's ID, the returner's ID,
                    and the result (1 if tackle successful, 0 otherwise)
    '''
    returner_id_list = []
    tackler_id_list = []
    tackle_results = []

    i = 0
    for index, row in merged_plays_pff.iterrows():
        returner = row["returnerId"]
        missed_tacklers = row["missedTackler"]
        successful_tackler = row["tackler"]
        game = row["gameId"]
        play = row["playId"]
        if missed_tacklers != "None":
            missed_tacklers_list = missed_tacklers.split(";")
            for tackler in missed_tacklers_list:
                tackler = tackler.strip(" ")
                try:
                    tackler_nfl_id = df_jerseyMap[(df_jerseyMap["gameId"] == game) & (df_jerseyMap["teamJersey"] == tackler)]["nflId"].values[0]
                except:
                    continue
                returner_id_list.append(returner)
                tackler_id_list.append(int(tackler_nfl_id))
                tackle_results.append(0)
        if successful_tackler != "None":
            try:
                tackler_nfl_id = df_jerseyMap[(df_jerseyMap["gameId"] == game) & (df_jerseyMap["teamJersey"] == successful_tackler)]["nflId"].values[0]
            except:
                continue
            tackler_id_list.append(int(tackler_nfl_id))
            returner_id_list.append(returner)
            tackle_results.append(1)
             
    matchup_dict = {"returner_id": returner_id_list, "tackler_id": tackler_id_list, "results": tackle_results}
    matchup_df = pd.DataFrame.from_dict(matchup_dict)

    return matchup_df


def calc_probability(rating1, rating2):
    '''
    Calculates the probability of player 1 winning a matchup
    '''
    prob = 1 / (1 + 10 ** ((rating2 - rating1) / 400))
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


def update_elo_weighted(returner_rating, tackler_rating, result, k, prob_tackle):
    '''
    Based on the results of a matchup, updates the elo ratings of the players
    Inputs:
        returner_rating: (float) returner's current rating
        tackler_rating: (float) tackler's current rating
        prob_tackle: (float) general probability that any given tackle is made
        k: (int) a constant 
        prob_tackle: (float) general probability that any given tackle is made
    Returns: 
        returner_rating, tackler_rating: Updated ratings
    '''
    returner_prob, tackler_prob = weighted_probs(returner_rating, tackler_rating, prob_tackle)
      
    if result == 1:
        tackler_rating = tackler_rating + k * (1 - tackler_prob)
        returner_rating = returner_rating + k * (0 - returner_prob)
    else:
        tackler_rating = tackler_rating + k * (0 - tackler_prob)
        returner_rating = returner_rating + k * (1 - returner_prob)
    
    return returner_rating, tackler_rating


def establish_elo_scores_weighted(matchup_df, k, prob_tackle):
    '''
    Calculate the elo scores for players
    Inputs:
        matchup_df: df of all the matchup. Each row contains the tackler's ID, the returner's ID,
                    and the result (1 if tackle successful, 0 otherwise)
        k: (int) a constant 
        prob_tackle: (float) general probability that any given tackle is made
    Returns:
        returners_scores: (dict) mapping returner's id to their rating
        returners_scores: (dict) mapping returner's id to their rating
    '''
    returners_scores = {}
    tacklers_scores = {}
    # Initiall set all scores to 1000
    for index, row in matchup_df.iterrows():
        returners_scores[row["returner_id"]] = 1000
        tacklers_scores[row["tackler_id"]] = 1000
        
    i = 0
    for index, row in matchup_df.iterrows():
        returner_id = row["returner_id"]
        tackler_id = row["tackler_id"]
        result = row["results"]
        returner_score = returners_scores[returner_id]
        tackler_score = tacklers_scores[tackler_id]
        new_returner_score, new_tackler_score = update_elo_weighted(returner_score, tackler_score, result, k, prob_tackle)
        
        returners_scores[returner_id] = new_returner_score
        tacklers_scores[tackler_id] = new_tackler_score

    return returners_scores, tacklers_scores


def combine_scores_to_df(returners_scores, tacklers_scores):
    '''
    Convert elo rating dictionaries to a dataframe which shows the tackle ratings
    and returner ratings for each player. Saves this to csv
    Inputs:
        returners_scores: (dict) mapping returner's id to their rating
        returners_scores: (dict) mapping returner's id to their rating
    Returns:
        all_elos_df: dataframe which shows the tackle ratings
    and returner ratings for each player
    '''
    returners_elos_df = pd.DataFrame.from_dict(returners_scores, orient="index")
    tacklers_elos_df = pd.DataFrame.from_dict(tacklers_scores, orient="index")

    returners_elos_df.reset_index(level=0, inplace=True)
    tacklers_elos_df.reset_index(level=0, inplace=True)

    returners_elos_df.rename(columns={"index": "nflId", 0: "elo_rating"}, inplace=True)
    tacklers_elos_df.rename(columns={"index": "nflId", 0: "elo_rating"}, inplace=True)

    all_elos_df = pd.merge(returners_elos_df, tacklers_elos_df, how='outer', on='nflId')
    all_elos_df.rename(columns={"elo_rating_x": "returner_elo", "elo_rating_y": "tackler_elo"}, inplace=True)
    all_elos_df.to_csv("elo.csv", index=False)

    return all_elos_df

def calculate_and_save_elos():
    '''
    Calculate elo rating for returners and tacklers. Save to csv
    Return:
        all_elos_df: dataframe which shows the tackle ratings 
    '''
    df_tracking = pd.concat([df_track_2018, df_track_2019, df_track_2020])
    kick_plays_df = process_play_data(df_plays)
    pff_tackle_df = process_pff_data(df_pff)
    
    df_jerseyMap = create_jersey_map(df_tracking, df_games)
    merged_plays_pff = pd.merge(pff_tackle_df, kick_plays_df, how='inner')
    merged_plays_pff.fillna(value="None", inplace=True)

    matchup_df = get_matchups(merged_plays_pff, df_jerseyMap)

    returners_scores, tacklers_scores = establish_elo_scores_weighted(matchup_df, 32, 0.7321341759844434)
    all_elos_df = combine_scores_to_df(returners_scores, tacklers_scores)

    return all_elos_df


if __name__ == '__main__':
    all_elos_df = calculate_and_save_elos()





