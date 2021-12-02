from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, AvgPool1D, AvgPool2D, Reshape,
    Input, Activation, BatchNormalization, Dense, Add, Lambda, Dropout, LayerNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback, EarlyStopping
import tensorflow as tf

from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pickle
import argparse
import time
from pathlib import Path


min_idx_y = 94
max_idx_y = 154
num_classes_y = max_idx_y - min_idx_y + 1

input_channels_dict = {"comb_acc": 14, "comb_elo":11, "comb_force_momentum":14, "comb_momentum":10, "comb":10,
                           "kick_acc":14, "kick_elo":11, "kick_force_momentum":14, "kick_momentum":10, "kick":10,
                           "punt_acc":14, "punt_elo":11, "punt_force_momentum":14, "punt_momentum":10, "punt":10}


def crps(y_true, y_pred):
    '''
    Calculate Continuous Ranked Probability Score. This is the loss function for the model
    Acknowledgement: This function is directly copied from code written by Gordeev and Singer for their model.
                     https://www.kaggle.com/jccampos/nfl-2020-winner-solution-the-zoo
    Inputs:
        y_true: 2D array of size (batch_size, num_classes_y). Each row in the array
                is a one hot encoding of the true yards gained on the play
        y_pred: 2D array of size (batch_size, num_classes_y). Each row in the array
                is a the dicrete probability distribution predicted by the model
    Returns:
        loss: (float) calculated crps
    '''
    loss = K.mean(K.sum((K.cumsum(y_pred, axis = 1) - K.cumsum(y_true, axis=1))**2, axis=1))/199
    return loss


def get_conv_net(num_classes_y, input_channels=10):
    '''
    Create the model
    Acknowledgement: This function is directly copied from code written by Gordeev and Singer for their model.
                     https://www.kaggle.com/jccampos/nfl-2020-winner-solution-the-zoo
    Inputs:
        num_classes_y: The number of discrete yard values for which the model predicts probabilities. 
                       In otherwords, it's the size of the model's output array
        input_channels: The number of channels for the input tensor to the model
    Returns:
        model: (Tensorflow model mobel object)
    '''
    inputdense_players = Input(shape=(11,10,input_channels), name = "playersfeatures_input")
    
    X = Conv2D(128, kernel_size=(1,1), strides=(1,1), activation='relu')(inputdense_players)
    X = Conv2D(160, kernel_size=(1,1), strides=(1,1), activation='relu')(X)
    X = Conv2D(128, kernel_size=(1,1), strides=(1,1), activation='relu')(X)
    
    Xmax = MaxPooling2D(pool_size=(1,10))(X)
    Xmax = Lambda(lambda x1 : x1*0.3)(Xmax)

    Xavg = AvgPool2D(pool_size=(1,10))(X)
    Xavg = Lambda(lambda x1 : x1*0.7)(Xavg)

    X = Add()([Xmax, Xavg])
    X = Lambda(lambda y : K.squeeze(y,2))(X)
    X = BatchNormalization()(X)
    
    X = Conv1D(160, kernel_size=1, strides=1, activation='relu')(X)
    X = BatchNormalization()(X)
    X = Conv1D(96, kernel_size=1, strides=1, activation='relu')(X)
    X = BatchNormalization()(X)
    X = Conv1D(96, kernel_size=1, strides=1, activation='relu')(X)
    X = BatchNormalization()(X)
    
    Xmax = MaxPooling1D(pool_size=11)(X)
    Xmax = Lambda(lambda x1 : x1*0.3)(Xmax)

    Xavg = AvgPool1D(pool_size=11)(X)
    Xavg = Lambda(lambda x1 : x1*0.7)(Xavg)

    X = Add()([Xmax, Xavg])
    X = Lambda(lambda y : K.squeeze(y,1))(X)
    
    X = Dense(96, activation="relu")(X)
    X = BatchNormalization()(X)

    X = Dense(256, activation="relu")(X)
    X = LayerNormalization()(X)
    X = Dropout(0.3)(X)

    outsoft = Dense(num_classes_y, activation='softmax', name = "output")(X)

    model = Model(inputs = [inputdense_players], outputs = outsoft)
    return model


def run_model(X_tensor, y_tensor, n_epochs, batch_size, lr, validation_data, decay_steps, input_channels=10, early_stop=False, data=''):
    '''
    Creates a new model and trains it.
    Inputs:
        X_tensor: Training X data of size (num_training_plays, 11, 10, input_channels)
        y_tensor: Training y data of size (num_training_plays, num_classes_y)
        n_epochs: (int) max number of epochs
        batch_size: (int)
        lr: (float) starting learning rate for lr scheduler
        vaidation: (tuple) of validation data (X_validation, y_validation)
        decay_steps: (int) Number of steps before learning rate is decayed
        input_channels: (int) The number of channels in the input X tensors
        early_stop: (boolean) True if want to use early stopping
        data: (str) Description of which input data type the model is training on
    Returns:
        model: (Tensorflow model mobel object) the trained model
    '''
    model = get_conv_net(num_classes_y, input_channels=input_channels)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr,
                                                              decay_steps=decay_steps,
                                                              decay_rate=0.9)
    print("decay_steps:", decay_steps)

    opt = Adam(learning_rate=lr_schedule)
    model.compile(loss=crps,
                  optimizer=opt)

    es = EarlyStopping(monitor='val_loss',
                        mode='min',
                        restore_best_weights=True,
                        verbose=1,
                        patience=50)

    if early_stop:
        print("We are early stopping \n")
        history = model.fit(X_tensor,
                          y_tensor, 
                          epochs=n_epochs,
                          batch_size=batch_size,
                          verbose=1,
                          validation_data=validation_data,
                          callbacks=[es])
    else:
        print("We are not early stopping \n")
        history = model.fit_generator(X_tensor,
                          y_tensor, 
                          epochs=n_epochs,
                          batch_size=batch_size,
                          verbose=1,
                          validation_data=validation_data)
        
    return model, history


def plot_loss(history, lr, i, decay_steps, data=''):
    '''
    Plot the training and validation loss
    Inputs:
        history: (Tensorflow History object) Contains information about loss
        batch_size: (int) batch size used to train model
        lr: (float) Starting learning rate for lr scheduler
        i: (int) which fold number of cross validation the loss is from
        data: (str) Description of which input data type the model is training on
    '''
    plt.title(f"Learning Curve {data} Learning Rate {lr}")
    plt.xlabel('Epoch')
    plt.ylabel('CRPS')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.legend()
    plt.savefig(f"model_plots/lr_cross_validation/{data}/plot_lr_{lr}_steps_{decay_steps}_{data}_{i}.png")
    plt.close()



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Kickoff/Punt Prediction Model')
    parser.add_argument('--input_desc', type=str, required=True)
    parser.add_argument('--epochs', type=int, required=False, default=400)
    parser.add_argument('--batch_size', type=int, required=False, default=2)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--decay_steps', type=int, required=True)
    parser.add_argument('--chunk', type=int, required=True)
    args = parser.parse_args()  

    
    input_channels = input_channels_dict[args.input_desc]

    # Load training data
    with open(f'input_tensors/X_tensor_{args.input_desc}_94_154_train.data', 'rb') as f:
        X_train_tensor = pickle.load(f)
    with open(f'input_tensors/y_tensor_{args.input_desc}_94_154_train.data', 'rb') as f:
        y_train_tensor = pickle.load(f).astype('float32')


    # Load arrays that stores which plays belong in the train split for each fold and
    # which plays belong in the validation set fot each fold
    data_desc = args.input_desc.split("_")[0]
    with open(f'hypertune_cross_splits/{data_desc}_train_splits.data', 'rb') as f:
        train_splits = pickle.load(f)
    with open(f'hypertune_cross_splits/{data_desc}_val_splits.data', 'rb') as f:
        val_splits = pickle.load(f)

    # Determine which folds we're evaluating
    if args.chunk == 1:
        split_indices = [0, 1, 2]
    else:
        split_indices = [3, 4]

    # If needed create new directories to store plots and validation losses
    Path(f"model_plots/lr_cross_validation/{args.input_desc}").mkdir(parents=True, exist_ok=True)
    Path(f"lr_cross_validation/{args.input_desc}").mkdir(parents=True, exist_ok=True)

    val_loss = []
    for i in split_indices:
        train_index = train_splits[i]
        val_index = val_splits[i]

        X_train, X_val = X_train_tensor[train_index], X_train_tensor[val_index]
        y_train, y_val = y_train_tensor[train_index], y_train_tensor[val_index]
        model, history = run_model(X_train_tensor, y_train_tensor, args.epochs, args.batch_size, args.lr, (X_val, y_val), args.decay_steps,
                                early_stop=True, input_channels=input_channels, data=args.input_desc)
        val_loss.append(min(model.history.history['val_loss']))

        plot_loss(history, args.lr, i, args.decay_steps, data=args.input_desc)


    with open(f'lr_cross_validation/{args.input_desc}/lr_cross_validation_{args.input_desc}_{args.lr}_{args.decay_steps}_{args.chunk}.data', 'wb') as f:
        pickle.dump(val_loss, f)

