import pickle
import numpy as np
import pandas as pd


def load_val_loss(file):
    with open(file, 'rb') as f:
        val_loss = pickle.load(f)
    return val_loss



descriptions = ["comb_acc", "comb_elo", "comb_force_momentum", "comb_momentum", "comb",
                "kick_acc", "kick_elo", "kick_force_momentum", "kick_momentum", "kick",
                "punt_acc", "punt_elo", "punt_force_momentum", "punt_momentum", "punt"]

df_values = []
val_loss_dict = {}

for desc in descriptions:
    row = [desc]
    loss = []
    for i in range(1, 5):
        loss += load_val_loss(f'model_cross_validation/{desc}/model_cross_validation_{desc}_{i}.data')
    print(loss)
    avg_loss = np.mean(np.array(loss))
    std = np.std(np.array(loss))
    row += loss
    row.append(avg_loss)
    row.append(std)
    df_values.append(row)
    val_loss_dict[desc] = loss
    #print(row)


df = pd.DataFrame(df_values, columns = ['input_desc', 'trial_1', 'trial_2', 'trial_3', 'trial_4', 'trial_5',
                                        'trial_6', 'trial_7', 'trial_8', 'trial_9', 'trial_10', 'mean', 'std'])
print(df)
df.to_csv("model_cross_val_df.csv", index=False)

with open("model_cross_val_dict.data", 'wb') as f:
    pickle.dump(val_loss_dict, f)