#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --time=24:00:00
#SBATCH --partition=amilan
#SBATCH --output=log_metadata-%j.out
#SBATCH --mail-type="ALL"
#SBATCH --mail-user="asum8093@colorado.edu"

module purge

module load anaconda
module load cuda/12.1.1
cd /scratch/alpine/asum8093/LongTailHengJi
conda activate py38-pt1131-cuda117
pip install datasets

echo "== This is the scripting step! =="

python contextual_features.py
echo "== End of Job =="