

from torch.utils.data import Dataset
from csv import reader
import torch
import random
import numpy as np


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
        
        if token in spec_tokens:
            #print(token)
            output_label.append(-1)
            output_token.append(token)
            continue
            
            
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




## Dataset used for pretraining with MLM
class EHRDataset(Dataset):
    def __init__(self, dataframe, max_len=64, tokenizer=None): # Should be changed 
        super(EHRDataset, self).__init__()
        
        self.age = dataframe['age']
        self.code = dataframe['diagnos_code']
        self.gender = dataframe['gender']
        self.dataframe = dataframe
        self.max_len = max_len
        self.ids = dataframe.subject_id
        self.tokenizer = tokenizer
        
        def _transform_data(data):
            
            """
            Transforms the records from containing arrays into a dictionary: 
            patient_record = {patientid1: records, patientid2: records, ...}
        
            """
            
            patient_records = {}
            
            for _, row in data.iterrows():
                patient_id, nradm, icd_codes, ndc_codes = row['subject_id'], len(list(row['hadm_id'])), list(row['diagnos_code']), list(row['medication_code'])
                age, gender = list(row['age']), list(row['gender'])
                admission_input, adm_age, adm_gender = [], [], []
                prior_values, prior_indicies = list(row['prior_values']), list(row['prior_indicies']) # Should the visits be seperated ?? 
                
                for i in range(nradm):
                    total_len = len(icd_codes[i]) +len(ndc_codes) + 1 # total tokenlength = nr_diagnose_codes + nr_medication_codes + SEP_token
                    admission_input.extend(icd_codes[i])
                    admission_input.extend(ndc_codes[i])
                    admission_input.extend(['[SEP]'])
                      
                    adm_age.extend([str(int(age[i]))]*total_len)
                    adm_gender.extend(gender*total_len)
                    
                patient_records[patient_id] = [admission_input, adm_age, adm_gender, prior_values, prior_indicies]
                
            return patient_records
        records = _transform_data(dataframe)    
        
        """
        The final representation after transforming should be:
        
        records[patientid1] -> [admissioninput (tokens), age, gender], where admissioninput will look like:
        
        admissioninput -> [code1, code2, medication1, medication2, tobacco, alcohol use, '[SEP]', ...]
        
        """
        self.patdata = records
        
        
    def __getitem__(self, index):
    
        patid = self.ids.iloc[index]
        
        record = self.patdata[patid]
                
        codes = record[0] # inputtokens
        
        age = record[1] # ages over the admissions
        
        gender = record[2]
        
        prior_values = record[3]
        prior_indicies = record[4]
        
        ## Cut sequence to the end of a visit
        codes = codes[(-self.max_len + 1):]
        idxlastsep = len(codes) - codes[::-1].index('[SEP]')
        codes = codes[:idxlastsep + 1]
        age = age[:idxlastsep + 1]
        gender = gender[:idxlastsep + 1]
        numbervisits = codes.count('[SEP]')
        
        prior_values = prior_values[:numbervisits]
        prior_indicies = prior_indicies[:numbervisits]
        prior_val, prior_indi = [], []
        for prior_values, prior_ind in zip(prior_values, prior_indicies):
            prior_val.extend(prior_values), prior_val.extend(['[SEP]'])
            prior_indi.extend(prior_ind), prior_indi.extend(['[SEP]'])
        
        voc = self.tokenizer.getVoc('code')
        prior_val = [voc[p] if ((p=='[SEP]') or (p=='[PAD]')) else p for p in prior_val]
        prior_in = []
        for indi in prior_indi:
            if indi not in ['[SEP]', '[PAD]']:
                code1, code2 = indi.split(',')
                prior_in.append(voc[code1])
                prior_in.append(voc[code2])
            else:
                prior_in.append(voc[indi])
                
        prior_indicies = prior_in
        
        if codes[0] != '[SEP]': # If the first token does not equal SEP, do not overwrite the token, instead insert CLS at the start
            codes = np.append(np.array(['[CLS]']), codes) # Append cls at the start, 
            age = np.append(np.array(age[0]), age)
            gender.insert(0, gender[0])
        else:
            codes[0] = '[CLS]' # If codes[0] equals SEP, overwrite it with a CLS token
        
        ## Create mask for padding tokens
        mask = np.ones(self.max_len)
        mask[len(codes):] = 0
        
        
        age = seq_padding(age, self.max_len) # Pad age 
        gender = seq_padding(gender, self.max_len) # Pad gender
        
        
        tokens, code, label = random_mask(codes, self.tokenizer) # Generate random masking
        
        tokens = seq_padding(tokens, self.max_len) # Pad tokens
        position = position_idx(tokens) # get position
        segment = index_seq(tokens) # index sequences
        
        codes = seq_padding(code, self.max_len) # Pad codes
        label = seq_padding(label, self.max_len, symbol=-1) # Pad labels with -1 as symbol, -1 will be ignored by cross entropy loss later
        
        
        age_ids = [voc[a] if a=='[PAD]' else int(a) for a in age] #self.tokenizer.convert_tokens_to_ids(age, 'age') # Convert age to ids
        gender_ids = self.tokenizer.convert_tokens_to_ids(gender, 'gender') # Convert gender to ids
        code_ids = self.tokenizer.convert_tokens_to_ids(codes, 'code') # Convert codes to ids
        label = self.tokenizer.convert_tokens_to_ids(label, 'code') # Convert labels to ids    
        
       # if len(prior_val) > len(prior_indicies):
        #    prior_indicies = seq_padding(prior_indicies, len(prior_val), symbol=-5)
        #else:
        #    prior_val = seq_padding(prior_val, len(prior_indicies), symbol=-5)
            
        test = [prior_val]    
        return (torch.LongTensor(age_ids), torch.LongTensor(gender_ids), torch.LongTensor(code_ids), torch.LongTensor(position), torch.LongTensor(segment), torch.LongTensor(mask), torch.LongTensor(label)), (torch.LongTensor(test))
          
    def __len__(self):
        return len(self.dataframe)
    
    
    
class EHRDatasetReadmission(Dataset):
    
    def __init__(self, dataframe, max_len=64, tokenizer=None):
        
        self.data = dataframe
        self.age = dataframe['age']
        self.alchoholabuse = dataframe['alcohol_abuse']
        self.tobbaco_abuse = dataframe['tobacco_abuse']
        self.code = dataframe['icd_code']
        self.gender = dataframe['gender']
        self.max_len = max_len
        self.ids = dataframe.subject_id
        self.tokenizer = tokenizer
        
        #self.prediction_task = prediction_task
        
        def _transform_data(data):
            patient_records = {} 
            for _, row in data.iterrows():
                patient_id, nradm, icd_codes = row['subject_id'], len(list(row['hadm_id'])), list(row['ccsr'])
                age, gender = list(row['age']), list(row['gender'])
                lab = list(row['label'])
                admission_input, adm_age, adm_gender, labels = [], [], [], []
                nvisits = 2
                if nradm < nvisits:
                    nvisits = nradm
                    
                for i in range(nvisits):
                    total_len = len(icd_codes[i]) + 1
                    admission_input.extend(icd_codes[i]) 
                    admission_input.extend(['[SEP]'])
                    
                    adm_age.extend([str(int(age[i]))]*total_len)
                    adm_gender.extend(gender*total_len)
                    
                labels = [lab[nvisits - 1]]
                    
                patient_records[int(patient_id)] = [admission_input, adm_age, adm_gender, labels]
                
            return patient_records
        records = _transform_data(dataframe) 
        self.patdata = records
        
    def __getitem__(self, index):
        
        patid = int(self.ids.iloc[index])
        
        record = self.patdata[patid]
        
        
        codes, age, gender, labels = record        
        
        age = age[(-self.max_len + 1):]
        gender = gender[(-self.max_len + 1):]
        codes = codes[(-self.max_len + 1):]
        
        if codes[0] != '[SEP]': 
            codes = np.append(np.array(['[CLS]']), codes)
            age = np.append(np.array(age[0]), age)
            gender.insert(0, gender[0])
        else:
            codes[0] = '[CLS]'
        
        mask = np.ones(self.max_len)
        mask[len(codes):] = 0
        
        
        age = seq_padding(age, self.max_len)
        gender = seq_padding(gender, self.max_len)
        label = seq_padding(labels, self.max_len, symbol=-1)
        tokens = codes 
        
        tokens = seq_padding(tokens, self.max_len)
        position = position_idx(tokens)
        segment = index_seq(tokens)
        
        
        codes = seq_padding(codes, self.max_len)
        
        age_ids = self.tokenizer.convert_tokens_to_ids(age, 'age')
        gender_ids = self.tokenizer.convert_tokens_to_ids(gender, 'gender')
        code_ids = self.tokenizer.convert_tokens_to_ids(codes, 'code')
        
        #if self.prediction_task!='readmission':
        #    label = self.tokenizer.convert_tokens_to_ids(label, 'label')
            
        return torch.LongTensor(age_ids), torch.LongTensor(gender_ids), torch.LongTensor(code_ids), torch.LongTensor(position), torch.LongTensor(segment), torch.LongTensor(mask), torch.LongTensor(label), torch.LongTensor([int(patid)])
        
    def __len__(self):
        return len(self.data)
    
    
    
## This dataset is used for the prediction tasks

class EHRDatasetCodePrediction(Dataset):
    
    def __init__(self, dataframe, max_len=64, tokenizer=None, prediction_task='ccsr'):
        
        self.data = dataframe
        self.age = dataframe['age']
        self.alchoholabuse = dataframe['alcohol_abuse']
        self.tobbaco_abuse = dataframe['tobacco_abuse']
        self.code = dataframe['icd_code']
        self.gender = dataframe['gender']
        self.max_len = max_len
        self.ids = dataframe.subject_id
        self.tokenizer = tokenizer
        
        self.prediction_task = prediction_task
        
        def _transform_data(data):
            patient_records = {} 
            for _, row in data.iterrows():
                patient_id, nradm, icd_codes = row['subject_id'], len(list(row['hadm_id'])), list(row['ccsr_traincodes'])
                age, gender = list(row['age']), list(row['gender'])
                if prediction_task == 'ccsr':
                    lab = list(row['ccsr_labels'])
                else:
                    lab = list(row['ndc_labels'])
                    
                    
                admission_input, adm_age, adm_gender, labels = [], [], [], []
                
                
                for i in range(nradm - 1):
                    total_len = len(icd_codes[i]) + 1
                    admission_input.extend(icd_codes[i]) 
                    admission_input.extend(['[SEP]'])
                    
                    adm_age.extend([str(int(age[i]))]*total_len)
                    adm_gender.extend(gender*total_len)
                    
                patient_records[int(patient_id)] = [admission_input, adm_age, adm_gender, lab]
                
            return patient_records
        records = _transform_data(dataframe) 
        self.patdata = records
        
    def __getitem__(self, index):
        
        patid = int(self.ids.iloc[index])
        
        record = self.patdata[patid]
        
        
        codes, age, gender, labels = record        
        
        age = age[(-self.max_len + 1):]
        gender = gender[(-self.max_len + 1):]
        codes = codes[(-self.max_len + 1):]
        
        if codes[0] != '[SEP]': 
            codes = np.append(np.array(['[CLS]']), codes)
            age = np.append(np.array(age[0]), age)
            gender.insert(0, gender[0])
        else:
            codes[0] = '[CLS]'
        
        mask = np.ones(self.max_len)
        mask[len(codes):] = 0
        
        
        age = seq_padding(age, self.max_len)
        gender = seq_padding(gender, self.max_len)
        labels = seq_padding(labels, self.max_len, symbol=-1)
        tokens = codes 
        
        tokens = seq_padding(tokens, self.max_len)
        position = position_idx(tokens)
        segment = index_seq(tokens)
        
        
        codes = seq_padding(codes, self.max_len)
        
        age_ids = self.tokenizer.convert_tokens_to_ids(age, 'age')
        gender_ids = self.tokenizer.convert_tokens_to_ids(gender, 'gender')
        code_ids = self.tokenizer.convert_tokens_to_ids(codes, 'code')
        
        labels = self.tokenizer.convert_tokens_to_ids(labels, 'label')
            
        return torch.LongTensor(age_ids), torch.LongTensor(gender_ids), torch.LongTensor(code_ids), torch.LongTensor(position), torch.LongTensor(segment), torch.LongTensor(mask), torch.LongTensor(labels), torch.LongTensor([int(patid)])
        
    def __len__(self):
        return len(self.data)
        
        
    
    
