#!/bin/bash

#SBATCH --job-name=ssK32
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --time=12:00:00
#SBATCH --output=slurm_sslK_32_%j.out

python py_jobs/sslK_32.py
