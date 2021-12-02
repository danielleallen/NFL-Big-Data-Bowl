import numpy as np
import matplotlib.pyplot as plt
import pickle

with open("model_cross_val_dict.data", 'rb') as f:
    val_dict = pickle.load(f)


labels = ["Benchmark", "Acceleration", "Momentum", "Force Momentum", "Elo"]
data_cats = ["comb", "kick", "punt"]
endings = ["", "_acc", "_momentum", "_force_momentum", "_elo"]



def get_box_inputs(data_cat, val_dict):
    '''
    Get all the values needed for a single box plot
    Inputs:
        data_cat: "comb", "kick" or "punt"
    '''
    vals = []
    for ending in endings:
        data_desc = data_cat + ending
        vals.append(val_dict[data_desc])
    return vals

def make_box_plot(vals_labels, comb_vals, data_cat):
    '''
    Create a box plot for a given play type
    '''
    if data_cat == "comb":
        title_desc = "Combination"
    elif data_cat == "kick":
        title_desc = "Kick"
    else:
        title_desc = "Punt"

    fig = plt.figure(figsize=(6,6))  # define the figure window
    ax  = fig.add_subplot(111)
    ax.boxplot(comb_vals)
    plt.title(f"Model Validations {title_desc}")
    plt.ylabel("CRPS")
    ax.set_xticklabels(vals_labels, fontsize=8)
    plt.savefig(f"boxplot_vals_{data_cat}.png")


for data_cat in data_cats:
    comb_vals = get_box_inputs(data_cat, val_dict)
    make_box_plot(labels, comb_vals, data_cat)
