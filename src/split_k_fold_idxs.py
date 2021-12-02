import numpy as np
import pickle
from sklearn.model_selection import KFold
import sys


def make_splits(data_desc, n_splits, folder):
    '''
    Gets indices for training and validation sets for each fold of cross-validation
    Inputs:
        data_desc: (str) either "kick", "punt", or "comb"
        n_splits: (int) number of folds in cross validation
        folder: (str) folder to save output to

    '''
    with open(f'input_tensors/X_tensor_{data_desc}_94_154_train.data', 'rb') as f:
        X_train_tensor = pickle.load(f)
    with open(f'input_tensors/y_tensor_{data_desc}_94_154_train.data', 'rb') as f:
        y_train_tensor = pickle.load(f).astype('float32')
        
    kf = KFold(n_splits=n_splits, shuffle=True)
    
    train_splits = []
    val_splits = []
    for train_index, val_index in kf.split(X_train_tensor):
        np.random.shuffle(train_index)
        np.random.shuffle(val_index)
        train_splits.append(train_index)
        val_splits.append(val_index)
    
    with open(f'{folder}/{data_desc}_train_splits.data', 'wb') as f:
        pickle.dump(train_splits, f)
    with open(f'{folder}/{data_desc}_val_splits.data', 'wb') as f:
        pickle.dump(val_splits, f)
    
    return train_splits, val_splits


if __name__ == '__main__':
    data_desc = sys.argv[1]
    n_splits = sys.argv[2]
    folder = sys.argv[3]
    
    make_splits(data_desc, n_splits, folder)
