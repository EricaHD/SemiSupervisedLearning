#!/bin/bash

#SBATCH --job-name=ssm_c
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=72GB
#SBATCH --time=48:00:00
#SBATCH --output=ssm_continue_%j.out

python py_jobs/ssl_shake_mini_continue.py
