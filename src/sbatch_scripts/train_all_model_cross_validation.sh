#!/bin/sh

# Declare an array of string with type
declare -a input_descs=("comb_acc" "comb_elo" "comb_force_momentum" "comb_momentum" "comb" "kick_acc" "kick_elo" "kick_force_momentum" "kick_momentum" "kick" "punt_acc" "punt_elo" "punt_force_momentum" "punt_momentum" "punt" )
declare -a chunks=("1" "2" "3" "4" )


# Iterate the string array using for loop
for input_desc in ${input_descs[@]}; do
    for chunk in ${chunks[@]}; do
        sbatch model_cross_validation.sh -i $input_desc -c $chunk
    done
done