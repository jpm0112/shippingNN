#!/bin/bash
#PBS -l nodes=1:ppn=4:gpus=2
#PBS -l walltime=168:00:00
#PBS -N tft_tuning
#PBS -o tft_tuning.log
#PBS -e tft_tuning.err
#PBS -q gpu

source /apps/profiles/modules_asax.sh.dyn
module load cuda/11.8.0
module load anaconda/3-2025.12
source activate cuda5
cd ~/shipping
python tuning_tft_multiple_gpu.py
