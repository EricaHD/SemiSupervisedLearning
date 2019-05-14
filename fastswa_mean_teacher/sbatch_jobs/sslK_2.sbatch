#!/bin/bash

#SBATCH --job-name=ssK2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --time=12:00:00
#SBATCH --output=slurm_sslK_2_%j.out

python py_jobs/sslK_2.py
