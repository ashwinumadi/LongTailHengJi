from datasets import load_dataset
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path


#folder_path = Path('data/fewnerd')

#folder_path.mkdir(parents=True, exist_ok=True)
#print(f"Folder created at: {folder_path}")


dataset = load_dataset("DFKI-SLT/few-nerd", "supervised")
training_dataset = dataset['train']
c = 0
with open('train.jsonl', 'w') as file:
    for item in training_dataset:
        file.write(json.dumps(item) + '\n')
        c+=1

testing_dataset = dataset['test']
c = 0
with open('test.jsonl', 'w') as file:
    for item in testing_dataset:
        file.write(json.dumps(item) + '\n')
        c+=1


dev_dataset = dataset['validation']
dataset_list = [dict(item) for item in dev_dataset]
c = 0
with open('dev.jsonl', 'w') as file:
    for item in testing_dataset:
        file.write(json.dumps(item) + '\n')
        c+=1