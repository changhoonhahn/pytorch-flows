#!/bin/bash
#SBATCH -J p_XgivenY
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=general
#SBATCH --time=11:59:59 
#SBATCH --export=ALL
#SBATCH --output=/tigress/chhahn/arcoiris/_p_XgivenY.o
#SBATCH --mail-type=all
#SBATCH --mail-user=chhahn@princeton.edu

now=$(date +"%T")
echo "start time ... $now"

module load anaconda3
conda activate torch-env 

python /home/chhahn/projects/packages/pytorch-flows/p_XgivenY.py
now=$(date +"%T")
echo "end time ... $now"

