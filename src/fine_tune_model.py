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


min_idx_y = 71
max_idx_y = 150
num_classes_y = max_idx_y - min_idx_y + 1

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


def get_conv_net(num_classes_y):
    '''
    Create the model
    Acknowledgement: This function is directly copied from code written by Gordeev and Singer for their model.
                     https://www.kaggle.com/jccampos/nfl-2020-winner-solution-the-zoo
    Inputs:
        num_classes_y: The number of discrete yard values for which the model predicts probabilities. 
                       In otherwords, it's the size of the model's output array
    Returns:
        model: (Tensorflow model mobel object)
    '''
    inputdense_players = Input(shape=(11,10,10), name = "playersfeatures_input")
    
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


def run_model(X_tensor, y_tensor, n_epochs, batch_size, lr, checkpoint_path, plot=False, data=''):
    '''
    Fine tune the model stored at checkpoint_path
    Inputs:
        X_tensor: Training X data of size (num_training_plays, 11, 10, 10)
        y_tensor: Training y data of size (num_training_plays, num_classes_y)
        n_epochs: (int) max number of epochs
        batch_size: (int)
        lr: (float) learning rate
        checkpoint_path: (str) path to folder where model checkpoints should be saved
        data: (str) Description of which input data type the model is training on
    Returns:
        model: (Tensorflow model mobel object) the fine tuned model
    ''' 
    model = get_conv_net(num_classes_y)

    opt = Adam(learning_rate=lr)
    model.compile(loss=crps,
                  optimizer=opt)
    
    model.load_weights(checkpoint_path)

    history = model.fit(X_tensor,
                      y_tensor, 
                      epochs=n_epochs,
                      batch_size=batch_size,
                      verbose=1,
                      validation_split=0.2)
    if plot:
        plot_loss(history, batch_size, lr, n_epochs, data)
        
    return model


def plot_loss(history, batch_size, lr, n_epochs, data=''):
    '''
    Plot the training and validation loss
    Inputs:
        history: (Tensorflow History object) Contains information about loss
        batch_size: (int) batch size used to train model
        lr: (float) Starting learning rate for lr scheduler
        n_epochs: (int) max epochs
        data: (str) Description of which input data type the model is training on
    '''
    plt.title(f"Learning Curve Batch Size {batch_size} Learning Rate {lr}")
    plt.xlabel('Epoch')
    plt.ylabel('CRPS Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.legend()
    plt.savefig(f"model_plots/plot_bs_{batch_size}_lr_{lr}_ep_{n_epochs}_{data}.pdf")
    plt.show()


if __name__ == '__main__':
    # Load Kickoff/Punt training data
    with open('input_tensors/X_tensor_comb_train.data', 'rb') as f:
        X_train_tensor = pickle.load(f)
    with open('input_tensors/y_tensor_comb_train.data', 'rb') as f:
        y_train_tensor = pickle.load(f).astype('float32')

    model = run_model(X_train_tensor, y_train_tensor, 100, 2, 1e-5, "model_checkpoint/rush.ckpt", plot=True, data='fine_tune_comb')