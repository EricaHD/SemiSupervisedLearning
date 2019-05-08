#!/bin/bash

#SBATCH --job-name=ss_c
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=72GB
#SBATCH --time=60:00:00
#SBATCH --output=slurm_ss_continue_%j.out

python py_jobs/ssl_shake_continue.py
