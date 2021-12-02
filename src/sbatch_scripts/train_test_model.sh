#!/bin/bash
#
#SBATCH --output=/home/dallen32/slurm/out/%j.%N.stdout
#SBATCH --error=/home/dallen32/slurm/out/%j.%N.stderr
#SBATCH --chdir=/home/dallen32/nfl_models
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=10000
#SBATCH --time=0-04:00

while getopts i:c:l:d: flag
do
    case "${flag}" in
        i) input_desc=${OPTARG};;
		c) input_channels=${OPTARG};;
		l) lr=${OPTARG};;
		d) decay_steps=${OPTARG};;
    esac
done



python3 model_variable.py --input_desc $input_desc --input_channels $input_channels --lr $lr --decay_steps $decay_steps