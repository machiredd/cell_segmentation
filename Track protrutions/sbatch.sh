#!/bin/sh

#SBATCH --job-name=t
#SBATCH --output=track_all3.out
#SBATCH --partition=gpu
#SBATCH --gres gpu:p100:1
#SBATCH --time=2-00:00:00

module use /home/exacloud/software/modules
module load cudnn/7.6-10.1
module load cuda/10.1.243

eval "$(conda shell.bash hook)"
conda activate em3

python track_final.py
