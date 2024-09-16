import numpy as np
from transformers import AutoTokenizer
import torch
from datasets import load_dataset
import pandas as pd

# Load the CoNLL-2003 dataset
dataset = load_dataset("DFKI-SLT/few-nerd", "supervised")

training_dataset = dataset['train']
label_list_coarse = dataset['train'].features['ner_tags'].feature.names
label_list_fine = dataset['train'].features['fine_ner_tags'].feature.names


# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('roberta-large', add_prefix_space=True)

print('START')
token_freq_mat = np.zeros((50265, 67))

entities = set()
entity_type = []
fine_entity_type = []
c = 0
# Sample rne', 'tokens': ['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.'], 'pos_tags': [22, 42, 16, 21, 35, 37, 16, 21, 7], 'chunk_tags': [11, 21, 11, 12, 21, 22, 11, 12, 0], 'ner_tags': [3, 0, 7, 0, 0, 0, 7, 0, 0]}
for item in training_dataset:
  tokens = item['tokens']
  tags = item['ner_tags']
  fine_tags = item['fine_ner_tags']
  for token in range(len(tokens)):
    if tokens[token] in tokenizer.vocab:
      word_tokenized = tokenizer.tokenize(tokens[token])
      word_ids = tokenizer.convert_tokens_to_ids(word_tokenized)
      token_freq_mat[word_ids[0]][fine_tags[token]] +=1
  c+=1
print(token_freq_mat)

print('START 2')
fewnerd_embedding_ids = []
unique = 0
for t in range(len(token_freq_mat)):
  flag = False
  for i in range(len(token_freq_mat[t])):
    if i!=0 and int(token_freq_mat[t][i]) != 0 and not flag:
      unique +=1
      flag = True
      fewnerd_embedding_ids.append(t)
print(fewnerd_embedding_ids)
print(unique)


print('START 3')
weighted_token_freq_mat = np.zeros((66, unique))
for fine_type in range(1, 67):
  all_tokens_per_type = token_freq_mat[:,fine_type]
  s = 0
  for each_token in all_tokens_per_type:
    s+=each_token
  for index_each_token in range(len(all_tokens_per_type)):
    if all_tokens_per_type[index_each_token] != 0:
      weighted_token_freq_mat[fine_type-1][fewnerd_embedding_ids.index(index_each_token)] = \
        all_tokens_per_type[index_each_token]/s

print(weighted_token_freq_mat)
weighted_token_freq_dict = {
    'embedding_ids': fewnerd_embedding_ids,
    'type_weight': weighted_token_freq_mat
}
print(weighted_token_freq_dict)

tensor_token_freq_mat = torch.tensor(token_freq_mat)
tensor_weighted_token_freq = {k: torch.tensor(v) for k, v in weighted_token_freq_dict.items()}

torch.save(tensor_token_freq_mat, 'fewnerd_token_freq_mat.th')
torch.save(tensor_weighted_token_freq, 'fewnerd_weighted_token_freq.th')