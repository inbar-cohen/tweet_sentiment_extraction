import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset, random_split
from transformers import BertTokenizer

train_df = pd.read_csv('./data/train.csv')
train_df.head()
print(train_df.head())
print(len(train_df))

# Check for missing data
print(train_df.isna().sum())

# Remove rows with missing data
train_df.dropna(axis=0, inplace=True)
train_df.reset_index(drop=True, inplace=True)

# A fairly balance dataset with neutral class being 3-4K more (might still need to do class weights / sampling)
train_df['sentiment'].value_counts()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
tokenizer.decode(tokenizer.encode(train_df['sentiment'][0], train_df['text'][0]))

encoded = tokenizer.encode_plus(train_df['sentiment'][0], train_df['text'][0], add_special_tokens=True, max_length=150,
                                pad_to_max_length=True, return_token_type_ids=True, return_attention_mask=True,
                                return_tensors='pt')

print(encoded['input_ids'])
print('------------------')
print(encoded['attention_mask'])
print('------------------')
print(encoded['token_type_ids'])

input_ids = []
attention_masks = []
token_type_ids = []

for i in range(len(train_df)):
    encoded = tokenizer.encode_plus(train_df['sentiment'][i], train_df['text'][i], add_special_tokens=True,
                                    max_length=150,
                                    pad_to_max_length=True, return_token_type_ids=True, return_attention_mask=True,
                                    return_tensors='pt')

    input_ids.append(encoded['input_ids'])
    attention_masks.append(encoded['attention_mask'])
    token_type_ids.append(encoded['token_type_ids'])

# concatenate all the elements into one tensor
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
token_type_ids = torch.cat(token_type_ids, dim=0)

