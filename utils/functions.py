from csv import reader
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch import nn
import random
import math

import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

def write_age_to_file(age_list, file_path) -> None:
    """ Function to write age vocabulary file """
    """ age_list will be 2d, where outer index will be patient, and inner index is the patient's visits"""
    
    all_ages = []
    
    ## [[a1, a2], [a1, a2], [a1, a2, a3, a4], [a1, a2, a3]]
    
    for patient_ages in age_list:
        all_ages.extend(patient_ages)
        
    all_agesnp = np.array(all_ages)
    np.save(file_path, all_agesnp)
    
    
def write_codes_to_file(code_list, file_path) -> None: 
    """ Function used for writing processing list_ which can contain either Snomed codes, ccssr codes, icd codes """
    """ This function will be integrated with additional features later.. such as writing multiple code formats into one file (diagnosis codes, medication codes)"""
    
    all_codes = []
    
    ## [[[d1, d2], [d3, d5]], [[d1, d2]]]
    
    for patient in code_list:
        for patient_visit in patient:
            if -1 not in patient_visit:
                all_codes.extend(patient_visit)
            
    all_codes = list(set(all_codes))
    all_codes = np.array(all_codes)
    np.save(file_path, all_codes)
    
    
    
def load_model(path, model):
    # load pretrained model and update weights
    pretrained_dict = torch.load(path)
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    return model




def index_seq(tokens, symbol='[SEP]'):
    """
    Used to tell which sentence a specific token belongs to
    """
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
    
    """
    Pad sequences with less length than max_len
    """
    token_len = len(tokens)
    
    sequence = []
    for i in range(max_len):
        if token_len > i:
            sequence.append(tokens[i])
            
        else:
            sequence.append(symbol)
            
    return sequence
            

def position_idx(tokens, symbol='[SEP]'):
    
    """
    Calculate in what order the tokens are arranged
    """
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
    
    """
    Used for creating masks which is during the MLM pretraining. 
    
    """
    output_label = []
    output_token = []
    spec_tokens = set(['[CLS]', '[SEP]']) # We want to mask tokens except the special tokens
    for i, token in enumerate(tokens):
        
        #if token in spec_tokens:
        #    output_label.append(-1)
        #    output_token.append(token)
        #    continue
            
            
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            # 80 % randomly change token to mask token
            if prob < 0.8:
                output_token.append(symbol)

            # 10% randomly change token to random token
            
            elif prob < 0.9:
                output_token.append(random.choice(list(tokenizer.getVoc('code').keys()))) # Randomly replace the token with another token 

            # -> rest 10% randomly keep current token

            # append current token to output
            output_label.append(token) # If 
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1) # If we don not mask, set the current token_label to -1 (-1 will be ignored in the crossentropy loss)
            output_token.append(token)

    return tokens, output_token, output_label

def create_prior_guide(code_maps, input_ids):
    
    dd, dp, dm, pp, pd, pm, mm, md, mp = code_maps
    prior_guide = []
    visit_1 = 0
    for token1 in input_ids:
        if token1 == '[SEP]':
            visit_1+=1
        visit_2 = 0
        for token2 in input_ids:
            if token2 == '[SEP]':
                visit_2 += 1
                
            visitdiff = abs(visit_2 - visit_1)
            
            comb = str(visitdiff) + ', ' + str(token1) + ',' + str(token2)
            if token1 == token2:
                value = 1   
            elif ((token1 in set(['[CLS]', '[SEP]', '[PAD]'])) or  (token2 in set(['[CLS]', '[SEP]', '[PAD]']))):
                value = 0
            elif ((token1 == '[MASK]') or (token2 == '[MASK]')):
                value = 0
            else:
                if comb in dd:
                    value = dd[comb]
                elif comb in dp:
                    value = dp[comb]
                elif comb in dm:
                    value = dm[comb]    
                elif comb in pp:
                    value = pp[comb]
                elif comb in pd:
                    value = pd[comb]
                elif comb in pm:
                    value = pm[comb]
                elif comb in mm:
                    value = mm[comb]
                elif comb in md:
                    value = md[comb]
                elif comb in mp:
                    value = mp[comb]
                else:
                    value = 0
            prior_guide.append(value)
            
    return prior_guide


def train_test_val_split(data, train_ratio=0.8, validation_ratio=0.10, test_ratio=0.10):
    
    train, test = train_test_split(data, test_size=1-train_ratio, random_state=42)
    validation, test = train_test_split(test, test_size=test_ratio/(test_ratio + validation_ratio), random_state=42) 
    
    return train, validation, test

def vis_encoder_latent(encoded_outputs, mask, epoch, path, labels=None):
    
    tsne_reducer = TSNE(2) 
    figure, axis = plt.subplots(len(encoded_outputs), figsize=(15,15))
    for idx, out in enumerate(encoded_outputs):
        averaged_layer_hidden_states = torch.div(out.sum(dim=1),mask.sum(dim=1,keepdim=True))
        encoded_reduced = tsne_reducer.fit_transform(averaged_layer_hidden_states.cpu().numpy())
        if labels:
            labels = labels.numpy().reshape(-1)
            df = pd.DataFrame.from_dict({'x':encoded_reduced[:,0],'y':encoded_reduced[:,1], 'label':labels})
            sns.scatterplot(data=df,x='x',y='y',hue='label', ax=axis[idx]).set(title='Encoder Layer: {}'.format(idx))
        else:
            df = pd.DataFrame.from_dict({'x':encoded_reduced[:,0],'y':encoded_reduced[:,1]})
            sns.scatterplot(data=df,x='x',y='y', ax=axis[idx]).set(title='Encoder Layer: {}'.format(idx))
            
    figure.suptitle('Encoded latent space at epoch:{}'.format(epoch), fontsize=16)        
    plt.savefig(path + "latent_space_epoch_{}.{}".format(epoch, 'png'))

    
def get_attention_scores(model, datapoint, layer_i, tokenizer, config):
    
    def transpose_for_scores(config, x):
        new_x_shape = x.size()[:-1] + (config.num_attention_heads, int(config.hidden_size / config.num_attention_heads))
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    
    tokenized_sequence = datapoint[2]
    
    
    segment_ids = datapoint[4]
    
    outputs_query= []
    outputs_key= []
    
    def hook_query(module, input, output):
        #print ('in query')
        outputs_query.append(output)

    def hook_key(module, input, output):
        #print ('in key')
        outputs_key.append(output)

    model.bert.encoder.layer[layer_i].attention.self.query.register_forward_hook(hook_query)
    model.bert.encoder.layer[layer_i].attention.self.key.register_forward_hook(hook_key)
    l=model(tokenized_sequence,segment_ids)
    
    query_layer = transpose_for_scores(config,outputs_query[0])
    key_layer = transpose_for_scores(config,outputs_key[0])
    
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    attention_scores = attention_scores / math.sqrt(int(config.hidden_size / config.num_attention_heads))
    attention_probs = nn.Softmax(dim=-1)(attention_scores)
    
    return attention_probs, tokenized_sequence
    

