#!/bin/bash

#SBATCH --job-name=ssn2
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=84GB
#SBATCH --time=48:00:00
#SBATCH --output=slurm_ssl2_new_%j.out

python py_jobs/ssl2_new.py
