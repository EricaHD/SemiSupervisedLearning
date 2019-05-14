#!/bin/bash

#SBATCH --job-name=ssl_shake_mini_augment
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --output=slurm_ssl_shake_mini_augment_%j.out

module purge
module load cuda/9.0.176
module load cudnn/9.0v7.0.5
module load python3/intel/3.6.3
source /scratch/jtb470/SemiSupervisedLearning/env/bin/activate

python py_jobs/ssl_shake_mini_augment.py
