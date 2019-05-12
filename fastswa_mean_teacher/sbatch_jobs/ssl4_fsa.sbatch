#!/bin/bash

#SBATCH --job-name=ssf4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=72GB
#SBATCH --time=48:00:00
#SBATCH --output=slurm_ssl4_fsa_%j.out

python py_jobs/ssl4_fsa.py
