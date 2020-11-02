#!/bin/bash

# Request a GPU partition node and access to 1 GPU
#SBATCH -p gpu --gres=gpu:1

# Request 1 CPU core
#SBATCH -n 1
#SBATCH -t 12:00:00

module load cuda

module load python/3.7.4
module load pytorch/1.3.1

python train_posenet.py >> ccv_output.txt
