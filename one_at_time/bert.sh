#!/bin/bash
#SBATCH --job-name=para
#SBATCH --time=4:00:00
#SBATCH --mem=2G

module load Python

source /data/$USER/.envs/four_room/bin/activate

echo Starting Python program
python3 main.py $*

deactivate