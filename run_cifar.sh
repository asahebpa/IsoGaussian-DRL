#!/bin/bash
#SBATCH --job-name=run
#SBATCH --output=run.txt
#SBATCH --error=run.txt
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem=20Gb
#SBATCH --gres=gpu:1 
#SBATCH -c 32

module load anaconda/3
conda activate deeprl
python ./cifar_sigreg.py --mlp_depth=medium --mlp_width=medium --use_ln --non_stationary --lambda_sig=0.0

