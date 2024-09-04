#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --time=00:10:00
#SBATCH --partition=aa100
#SBATCH --gres=gpu:1
#SBATCH --output=run_test-%j.out
#SBATCH --mail-type="ALL"
#SBATCH --mail-user="asum8093@colorado.edu"

module purge

module load anaconda
module load cuda/12.1.1
cd /scratch/alpine/asum8093/LongTailHengJi
conda activate py38-pt1131-cuda117

echo "== This is the scripting step! =="


CUDA_VISIBLE_DEVICES={0}, python run_train.py --gpu 0 --model-name roberta-large --log-dir log_fewnerd/roberta-large-vanilla/ --eval-method macro --run-method vanilla --max-length 256 --dataset fewnerd --min-epoch -1 --root data/fewnerd/ --n-class 67 --task-of-label entity --word-level

echo "== End of Job =="