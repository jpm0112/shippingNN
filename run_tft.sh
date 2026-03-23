#!/bin/bash
source /apps/profiles/modules_asax.sh.dyn
module load cuda/11.8.0
module load anaconda/3-2025.12
source activate cuda5
cd ~/shipping
python tuning_tft_multiple_gpu.py
