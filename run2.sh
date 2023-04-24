#!/bin/bash
module purge
module load Python/3.10.8-GCCcore-12.2.0

source $HOME/.envs/DRL_lab/bin/activate

python3 run.py $@

deactivate