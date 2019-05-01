#!/bin/bash

#SBATCH --job-name=ssl_shake_mini_sup
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --time=30:00:00
#SBATCH --output=slurm_ssl_shake_mini_sup_%j.out

python experiments/ssl_shake_mini_sup.py
