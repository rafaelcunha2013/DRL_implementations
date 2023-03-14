#!/bin/bash
#SBATCH --job-name=para
#SBATCH --time=30:00
#SBATCH --mem=1G

module load Python

source /data/$USER/.envs/four_room/bin/activate

echo Starting Python program
python3 summary_writer.py

deactivate