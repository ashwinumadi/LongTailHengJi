#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --time=24:00:00
#SBATCH --partition=blanca-curc-gpu
#SBATCH --gres=gpu:1
#SBATCH --output=blanca_run_base_model-%j.out
#SBATCH --mail-type="ALL"
#SBATCH --mail-user="asum8093@colorado.edu"

module purge

module load anaconda
module load cuda/12.1.1
cd /scratch/alpine/asum8093/LongTailHengJi
conda activate py38-pt1131-cuda117

echo "== This is the scripting step! =="

python run_train.py --gpu 0 --model-name roberta-large --log-dir log_fewnerd/roberta-large-vanilla/ --eval-method macro --run-method vanilla --max-length 256 --dataset fewnerd --min-epoch -1 --root data/fewnerd/ --n-class 67 --task-of-label entity --word-level

source_file_path="./log_fewnerd/roberta-large-vanilla/log_fewnerd/roberta-large-vanilla/model.42"
destination_folder_path="./log_fewnerd/roberta-large-vanilla/"

# Move the file
mv "$source_file_path" "$destination_folder_path"
echo "== End of Job =="