
from torch.utils.data import Dataset
from csv import reader
import torch
import random
import numpy as np

def index_seq(tokens, symbol='[SEP]'):
    flag = 0
    seg = []

    for token in tokens:
        if token == symbol:
            seg.append(flag)
            if flag == 0:
                flag = 1
            else:
                flag = 0
        else:
            seg.append(flag)
    return seg


def seq_padding(tokens, max_len, symbol='[PAD]'): # maybe implement for unknown token ?
    
    token_len = len(tokens)
    
    sequence = []
    for i in range(max_len):
        if token_len > i:
            sequence.append(tokens[i])
            
        else:
            sequence.append(symbol)
            
    return sequence
            

def position_idx(tokens, symbol='[SEP]'):
    pos = []
    flag = 0

    for token in tokens:
        if token == symbol:
            pos.append(flag)
            flag += 1
        else:
            pos.append(flag)
    return pos



def random_mask(tokens, tokenizer, symbol='[MASK]'):
    output_label = []
    output_token = []
    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                output_token.append(symbol)

            # 10% randomly change token to random token
            
            elif prob < 0.9:
                output_token.append(random.choice(list(tokenizer.getVoc('code').keys()))) # This row is for randomly choose a token

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later
            output_label.append(token) # Unclear
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)
            output_token.append(token)

    return tokens, output_token, output_label


            
class EHRDataset(Dataset):
    def __init__(self, dataframe, max_len=64, tokenizer=None):
        super(EHRDataset, self).__init__()
        
        self.age = dataframe.dropna()['age']
        self.code = dataframe.dropna()['icd_code'] 
        self.max_len = max_len
        
        self.tokenizer = tokenizer
        
        
    def __getitem__(self, index):
        
       # print(index)
        age = self.age[index].replace('SEP', '[SEP]').split(',')[(-self.max_len):]
        codes = self.code[index].replace('SEP', '[SEP]').split(',')[(-self.max_len):]
        
        # Start by tokenizing the visits
        # Then add special tokens
        # Transform the data, when index is in into:
        
        mask = np.ones(self.max_len)
        mask[len(codes):] = 0
        
        
        age.insert(0, '[CLS]')
        codes.insert(0, '[CLS]')
        
        age = seq_padding(age, self.max_len)
        
        tokens, code, label = random_mask(codes, self.tokenizer)
        
        tokens = seq_padding(tokens, self.max_len)
        position = position_idx(tokens)
        segment = index_seq(tokens)
        
        
        codes = seq_padding(codes, self.max_len)
        label = seq_padding(label, self.max_len, symbol=-1)
        
        age_ids = self.tokenizer.convert_tokens_to_ids(age)
        code_ids = self.tokenizer.convert_tokens_to_ids(codes)
        label = self.tokenizer.convert_tokens_to_ids(label)
        
        return torch.LongTensor(age_ids), torch.LongTensor(code_ids), torch.LongTensor(position), torch.LongTensor(segment), \
    torch.LongTensor(mask), torch.LongTensor(label)
        
        
    def __len__(self):
        return len(self.code)
        
        
        
        
        
    
    
