#!/bin/bash
#SBATCH -J p_Y
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=general
#SBATCH --time=11:59:59 
#SBATCH --export=ALL
#SBATCH --output=_p_Y.o
#SBATCH --mail-type=all
#SBATCH --mail-user=chhahn@princeton.edu

now=$(date +"%T")
echo "start time ... $now"

module load anaconda3
conda activate torch-env 

python /home/chhahn/projects/packages/pytorch-flows/p_Y.py
now=$(date +"%T")
echo "end time ... $now"

