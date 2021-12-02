import pickle
import numpy as np
from sklearn.model_selection import train_test_split

def get_train_test_split_idxs(data_desc, size):
    '''
    We want all models training on the same play-type to have the same train test split. This function
    generates the indices for the train and test sets
    Inputs:
        data_desc: Either "kick" or "punt"
        size: Number of plays in dataset
    Returns:
        X_train, X_test, y_train, y_test: 1D arrays that store the indices
    '''
    X_train, X_test, y_train, y_test = train_test_split(np.arange(size), np.arange(size), test_size=0.1)
    with open(f'train_test_idxs/X_{data_desc}_train_idxs.data', 'wb') as f:
        pickle.dump(X_train, f)
    with open(f'train_test_idxs/X_{data_desc}_test_idxs.data', 'wb') as f:
        pickle.dump(X_test, f)
    with open(f'train_test_idxs/y_{data_desc}_train_idxs.data', 'wb') as f:
        pickle.dump(y_train, f)
    with open(f'train_test_idxs/y_{data_desc}_test_idxs.data', 'wb') as f:
        pickle.dump(y_test, f)

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    get_train_test_split_idxs("kick", 2764)
    get_train_test_split_idxs("punt", 2259)