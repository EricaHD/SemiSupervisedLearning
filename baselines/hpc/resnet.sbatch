#!/bin/bash

#SBATCH --job-name=resnet-training
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100GB
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END
#SBATCH --mail-user=jtb470@nyu.edu
#SBATCH --output=slurm_resnet_%j.out

module purge
unset XDG_RUNTIME_DIR
if [ "$SLURM_JOBTMP" != "" ]; then
    export XDG_RUNTIME_DIR=$SLURM_JOBTMP
fi

module load python3/intel/3.6.3
module load cuda/9.0.176
module load cudnn/9.0v7.3.0.29
source /scratch/jtb470/SemiSupervisedLearning/env/bin/activate

python models/resnet.py

