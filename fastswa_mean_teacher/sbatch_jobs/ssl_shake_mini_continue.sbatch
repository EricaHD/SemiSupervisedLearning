#!/bin/bash

#SBATCH --job-name=ssm_c
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=72GB
#SBATCH --time=48:00:00
#SBATCH --output=slurm_ssm_continue_%j.out

module purge
module load cuda/9.0.176
module load cudnn/9.0v7.0.5
module load python3/intel/3.6.3
source /scratch/jtb470/SemiSupervisedLearning/env/bin/activate

python py_jobs/ssl_shake_mini_continue.py
