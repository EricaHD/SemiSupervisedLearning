#!/bin/bash

#SBATCH --job-name=ssl_shake_mini_augment
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --time=48:00:00
#SBATCH --output=ssl_shake_mini_augment_%j.out

python experiments/ssl_shake_mini_augment.py
