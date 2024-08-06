# Generating Vanilla Code

!CUDA_VISIBLE_DEVICES={0}, python run_train.py --gpu 0 --model-name roberta-large --log-dir log_fewnerd/roberta-large-vanilla/ --eval-method macro --run-method vanilla --max-length 256 --dataset fewnerd --min-epoch -1 --root data/fewnerd/ --n-class 169 --task-of-label entity --word-level