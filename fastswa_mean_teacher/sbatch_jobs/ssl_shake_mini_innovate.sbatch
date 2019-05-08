#!/bin/bash

#SBATCH --job-name=ssm_i
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --time=30:00:00
#SBATCH --output=slurm_ssm_innovate_%j.out

python py_jobs/ssl_shake_mini_innovate.py
