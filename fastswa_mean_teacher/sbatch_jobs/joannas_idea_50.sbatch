#!/bin/bash

#SBATCH --job-name=joi_50
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=72GB
#SBATCH --time=60:00:00
#SBATCH --output=slurm_joanna_50_%j.out

module purge
module load cuda/9.0.176
module load cudnn/9.0v7.0.5
module load python3/intel/3.6.3
source /scratch/jtb470/SemiSupervisedLearning/env/bin/activate

python py_jobs/joannas_idea_50.py
