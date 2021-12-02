#!/bin/sh

# Declare an array of string with type
declare -a input_descs=("comb_acc" "comb_elo" "comb_force_momentum" "comb_momentum" "comb" "kick_acc" "kick_elo" "kick_force_momentum" "kick_momentum" "kick" "punt_acc" "punt_elo" "punt_force_momentum" "punt_momentum" "punt" )
declare -a lrs=("1e-4" "1e-5" )
declare -a decay_steps=("10000" "50000" "100000" )


# Iterate the string array using for loop
for input_desc in ${input_descs[@]}; do
    for lr in ${lrs[@]}; do
        for decay_step in ${decay_steps[@]}; do
            sbatch train_model.sh -i $input_desc -l $lr -d $decay_step
        done
    done
done