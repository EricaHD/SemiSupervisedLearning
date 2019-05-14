#!/bin/bash

#SBATCH --job-name=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --output=slurm_main_%j.out
#SBATCH --mail-type=END
#SBATCH --mail-user=ehd255@nyu.edu

module purge
module load python3/intel/3.6.3
module load cuda/9.0.176
module load cudnn/9.0v7.0.5
source /home/ehd255/pyenv/py3.6.3/bin/activate

python /home/ehd255/SemiSupervisedLearning/main.py
