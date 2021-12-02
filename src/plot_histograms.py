import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

kick_play_track_df = pd.read_csv("play_track_kickoff.csv")
punt_play_track_df = pd.read_csv("play_track_punt.csv")

return_yards_kicks = kick_play_track_df.groupby(["gameId", "playId"])["kickReturnYardage"].mean().values
return_yards_punts = punt_play_track_df.groupby(["gameId", "playId"])["kickReturnYardage"].mean().values
all_return_yards = np.concatenate((return_yards_kicks, return_yards_punts))

bins = [i for i in range(-15, 100, 1)]

plt.hist(return_yards_kicks, bins=bins)
plt.title("Yards Gained on Kickoffs")
plt.xlabel("Yards Gained on Return")
plt.savefig("histograms/kick_hist.png")

plt.hist(return_yards_punts, bins=bins)
plt.title("Yards Gained on Punts")
plt.xlabel("Yards Gained on Return")
plt.savefig("histograms/punt_hist.png")

plt.hist(all_return_yards, bins=bins)
plt.title("Yards Gained on Kicks and Punts")
plt.xlabel("Yards Gained on Return")
plt.savefig("histograms/comb_hist.png")

