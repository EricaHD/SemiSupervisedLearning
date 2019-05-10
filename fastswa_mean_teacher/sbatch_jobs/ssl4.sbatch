#!/bin/bash

#SBATCH --job-name=ss4
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=84GB
#SBATCH --time=48:00:00
#SBATCH --output=slurm_ssl4_%j.out

python py_jobs/ssl4.py
