import numpy as np
from transformers import AutoTokenizer
import torch
from datasets import load_dataset
import pandas as pd
from transformers import RobertaTokenizer, RobertaModel
from datasets import load_dataset

# Load the CoNLL-2003 dataset
dataset = load_dataset("DFKI-SLT/few-nerd", "supervised")

training_dataset = dataset['train']


# Load pre-trained RoBERTa model and tokenizer
model_name = "roberta-large"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaModel.from_pretrained(model_name)

# Ensure the model is in evaluation mode
model.eval()

# Define the function to get contextual features
def get_contextual_features(sentence, model, tokenizer):
    tokens = tokenizer.tokenize(sentence)
    input_ids = tokenizer.encode(sentence, return_tensors='pt')

    seq_len = input_ids.size(1)
    hidden_dim = model.config.hidden_size

    # Initialize the tensor to store contextual features
    contextual_features = torch.zeros((seq_len, hidden_dim))

    # Iterate over each token position
    for i in range(seq_len):
        # Mask the token at position i
        masked_input_ids = input_ids.clone()
        masked_input_ids[0, i] = tokenizer.mask_token_id

        # Get the output of the model
        with torch.no_grad():
            outputs = model(masked_input_ids)

        # Extract the hidden state of the masked token
        contextual_features[i] = outputs.last_hidden_state[0, i]

    return contextual_features

# Example to process a single training instance
# In practice, you should iterate over the entire dataset
import os

# Create the directory if it doesn't exist
os.makedirs('contextual_features/', exist_ok=True)

for i, example in enumerate(training_dataset):
    sentence = example['tokens']
    sentence = ' '.join(sentence)  # Convert list of tokens to a single string

    # Get the contextual features for the sentence
    features = get_contextual_features(sentence, model, tokenizer)

    # Save the tensor to a file
    torch.save(features, f'contextual_features/{i}')
