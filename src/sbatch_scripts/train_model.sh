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
#SBATCH --gpus=1

while getopts i:l:d: flag
do
    case "${flag}" in
        i) input_desc=${OPTARG};;
		l) lr=${OPTARG};;
		d) decay_steps=${OPTARG};;
    esac
done



python3 lr_cross_validation.py --input_desc $input_desc --lr $lr --decay_steps $decay_steps