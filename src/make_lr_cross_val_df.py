import pickle
import numpy as np
import pandas as pd
from os import listdir



def load_val_loss(file):
    with open(file, 'rb') as f:
        val_loss = pickle.load(f)
    return val_loss



descriptions = ["comb_acc", "comb_elo", "comb_force_momentum", "comb_momentum", "comb",
                "kick_acc", "kick_elo", "kick_force_momentum", "kick_momentum", "kick",
                "punt_acc", "punt_elo", "punt_force_momentum", "punt_momentum", "punt"]
lrs = ["0.0001", "1e-05"]
decay_steps=["10000", "50000", "100000"]




df_values = []

for desc in descriptions:
    files = listdir(f"lr_cross_validation/{desc}")
    for lr in lrs:
        for decay_step in decay_steps:
            if f"lr_cross_validation_{desc}_{lr}_{decay_step}_1.data" in files:
                loss_chunk_1 = load_val_loss(f"lr_cross_validation/{desc}/lr_cross_validation_{desc}_{lr}_{decay_step}_1.data")
                loss_chunk_2 = load_val_loss(f"lr_cross_validation/{desc}/lr_cross_validation_{desc}_{lr}_{decay_step}_2.data")
                loss = loss_chunk_1 + loss_chunk_2
            else:
                try:
                    loss = load_val_loss(f"lr_cross_validation/{desc}/lr_cross_validation_{desc}_{lr}_{decay_step}.data")
                except:
                    continue

            avg_loss = np.mean(np.array(loss))
            std = np.std(np.array(loss))
            row = [desc, lr, decay_step] + loss
            row.append(avg_loss)
            row.append(std)
            df_values.append(row)


df = pd.DataFrame(df_values, columns = ['input_desc', 'lr', 'decay_step', 'trial_1', 'trial_2', 'trial_3',
                                        'trial_4', 'trial_5', 'mean', 'std'])
print(df)
df.to_csv("lr_cross_val_df.csv", index=False)





