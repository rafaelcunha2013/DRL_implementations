#!/bin/bash
#SBATCH --job-name=para
#SBATCH --time=20:00:00
#SBATCH --mem=2G

module load Python

source /data/$USER/.envs/four_room/bin/activate

echo Starting Python program
python3 pytorch_ddqn_mult.py $*

deactivate