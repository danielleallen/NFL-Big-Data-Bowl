The /src directory contains the source code for this project. All files/scripts in this directory 
have been written by me, with the exception that a few functions and snippets of code were written
by other source. I have cited these sources in the docstrings of their corresponding functions

The /process_rushing folder contains the files that were solely written by Gordeev and Singer for their model
https://www.kaggle.com/jccampos/nfl-2020-winner-solution-the-zoo



Here I describe the files in the /src directory 


------------------------- Data Processing and Creating Input Tensors ------------------------------------
NOTE: None of these files take command line arguments so you can just run: python3 <filename>

process_data.py
- Number of Lines: 142
- Cleans and Processes the data. Creates dataframe that contains all the data needed for the benchmark model where each row contains a single player's tracking data for a given play. The processed kick plays are stored in play_track_kick.csv and the punt plays are stored in play_track_punt.csv. All of the following Data Processing/Inport Tensor Creators rely on these csvs.

create_benchmark_input_tensors.py
- Number of Lines: 209
- Creates the input X and y train and test tensors for the benchmark model and saves them as pickle files in /input_tensors directory

calculate_elo.py
- Number of Lines: 265
- Calculates the elo ratings for players. Saves the results in a csv

elo_inputs.py
- Number of Lines: 247
- Creates the input X and y train and test tensors for the elo model and saves them as pickle files in /input_tensors directory

momentum_inputs.py
- Number of Lines: 227
- Creates the input X and y train and test tensors for the momentum model and saves them as pickle files in /input_tensors directory

force_momentum_inputs.py
- Number of Lines: 230
- Creates the input X and y train and test tensors for the force momentum model and saves them as pickle files in /input_tensors directory

acceleration_inputs.py
- Number of Lines: 205
- Creates the input X and y train and test tensors for the acceleration model and saves them as pickle files in /input_tensors directory

process_data_first_contact.py
- Number of Lines: 250
- Processes data to find yards to first contact for each play of interest. From its generated dataframe, creates the X and y train and test tensors needed to train the benchmark model to predict yards to first contact.


-------------------------------------------- Splitting the Data ---------------------------------------------
Since I wished to compare models that were trained on each play type, I needed to make sure they had the same plays in their train and test sets, as well as in their folds for cross validations.
Additionally, because of time limits on UChicago's AI cluster, cross validation within a single model had to be submitted in job chunks. This required knowing ahead of time which plays should be in the train and validation set for each fold

train_test_idxs.py
- Number of lines: 29
- Determined which indices of the kickoff plays should be in the train and test set, as well as which indices of the punt plays should be in the train and test set. The indices were saved in a pickle file and stored in the /train_test_idxs directory

split_k_fold_idxs.py
- Number of lines: 45
- For each of k folds, determines which plays should be in the train set and which should be in the validation set. These are then the splits used by a models training on a given play type.
- The function takes two command line arguments:
	- data_desc: Whether these splits are for "kick" or "punt"
	- n_splits: The number of train/validation splits to make (or how many folds)
	- folder: name of folder to save pickle files to


-------------------------------------------- Training the Models --------------------------------------------
Below are the three main scripts I used for training the models. They all take a combination of several of the folling
command line parameters. When writing the parameter arguments on the command line, you must use the flags

--input_desc: The model we want to train on.
			  Here is what you should write as the argument for each of the model types:
				  Benchmark Kickoffs -> kick
				  Acceleration Kickoffs -> kick_acc
				  Momentum Kickoffs -> kick_momentum
				  Force Momentum Kickoffs -> kick_force_momentum
				  Elo Kickoffs -> kick_elo
				  Benchmark Punts -> punt
				  Acceleration Punts -> punt_acc
				  Momentum Punts -> punt_momentum
				  Force Momentum Punts -> punt_force_momentum
				  Elo Punts -> punt_elo
				  Benchmark Combination -> comb
				  Acceleration Combination -> comb_acc
				  Momentum Combination -> comb_momentum
				  Force Momentum Combination -> comb_force_momentum
				  Elo Combination -> comb_elo

--input_channels: (int) Number of channels in model's X input channel. Default is 10
--epochs: (int) max number of epochs to train for
--batch_size: (int)
--lr: (float) starting learning rate for learning rate scheduler
--decay_steps: (int) Number of steps lr scheduler should take before decaying
--chunk: (int) For cross validation, the jobs on the UChicago's AI cluster would sometimes hit the max timelimit. I therefore
               had to break the task down into chunks where each chunk corresponded to certain folds. For example, for the 10-fold model cross validation, chunk 1 corresponded to folds 1-3, chunk 2 corresponded to folds 4-6, etc.

Files:

train_test_model.py
- Number of Lines: 203
- Trains a model, graphs the training and validation loss, and reports the test loss. This was used after I had tuned the hyper-parameters and performed 10-fold cross validation.
- Command Line Parameters:
	--input_desc (required)
	--input_channels (optional but default set at 10)
	--epochs (optional but default set at 400)
	--batch_size (optional but default set at 2)
	--lr (optional but default set at 1e-4)
	--decay_steps (required)

lr_cross_validation.py
- Number of lines: 234
- Script used to run 5-Fold cross validation to determine optimal learning rate scheduler parameters
- Due to timelimits, the script only deals with one lr/decay rate at a time and it was up to me to create multiple jobs that execute this file in order to complete the cv
- Command Line Parameters:
	--input_desc (required)
	--epochs (optional but default set at 400)
	--batch_size (optional but default set at 2)
	--lr (required)
	--decay_steps (required)
	--chunk (required, must be either 1 or 2)

model_cross_validation.py
- Number of lines: 246
- Script used to run 10-Fold cross validation on model types
- Due to timelimits, the script only deals with one model at a time and it was up to me to create multiple jobs that execute this file in order to complete the cv
- Command Line Parameters:
	--input_desc (required)
	--epochs (optional but default set at 400)
	--chunk (required, between 1 and 4)



---------------------------------- Fine Tuning ----------------------------------------------
These are scripts I used when I attempted to fine tune the model by training it on rushing data first

train_rush_model.py
- Number of lines: 165
- Trains the benchmark model on the rushing dataset and continually saves the model

fine_tune_model.py
- Number of lines: 156
- Fine tunes the model trained on the rushing data with kickoff and punt data



----------------------------- Miscellaneous Files ----------------------------------------

plot_histograms.py
- Number of lines: 27
- Creates the histograms of yards gained

make_val_box_plot.py
- Number of lines: 49
- Creates box plots for validation loss from 10-fold cross validation

make_model_cross_val_df.py
- Number of lines: 42
- Reads in the pickle files that contain the losses from 10-fold cv and creates as csv and dictionary from the results

make_lr_cross_val_df.py
- Number of lines: 51
- Reads in the pickle files that contain the losses from 5-fold cv and creates a csv from the results


--------------------------- Additional Folders in /src -----------------------------------

/data_sets: Contains csvs from NFL

/hypertune_cross_splits: Contains the pickle files that dictate which plays belong in which fold for 5-fold lr cross validation

/input_tensors: Pickle files of the input tensors to model

/lr_cross_validation: Pickle files containing validations losses from 5-fold lr cross validation

/model checkpoint: Contains saved model trained on rushing data. Used for finetuning

/model_cross_splits: Contains the pickle files that dictate which plays belong in which fold for 10-fold cross validation

/model_cross_validation: Pickle files containing validations losses from 10-fold cross validation

/model_plots: Plots of losses generated during training

/sbatch_scipts: bash scripts used to run jobs no UChicago cluster

/train_test_idxs: Pickle files that contain which indices of the kickoff plays should be in the train and test set, as well as which indices of the punt plays should be in the train and test set.







