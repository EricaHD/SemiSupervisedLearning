#!/bin/bash

#SBATCH --job-name=dataset
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100GB
#SBATCH --time=10:00:00

module purge
module load cuda/9.0.176
module load cudnn/9.0v7.0.5
module load python3/intel/3.6.3
source /scratch/jtb470/SemiSupervisedLearning/env/bin/activate

python create_50_train.py
python create_50_val.py 
