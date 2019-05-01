#!/bin/bash

#SBATCH --job-name=0.01_ssl
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --time=30:00:00
#SBATCH --output=slurm_ssl_mt_shake_short_%j.out

module purge
module load cuda/9.0.176
module load cudnn/9.0v7.0.5
module load python3/intel/3.6.3
source /scratch/jtb470/SemiSupervisedLearning/env/bin/activate

python experiments/ssl_mt_shake_short_upsample_lower_lr.py
