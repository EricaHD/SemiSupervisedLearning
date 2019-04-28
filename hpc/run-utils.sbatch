#!/bin/bash

#SBATCH --job-name=utils
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --output=slurm_utils_%j.out
#SBATCH --mail-type=END
#SBATCH --mail-user=ehd255@nyu.edu

module purge
module load python3/intel/3.6.3
source /home/ehd255/pyenv/py3.6.3/bin/activate

python /home/ehd255/SemiSupervisedLearning/utils.py
