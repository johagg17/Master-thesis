
from torch.utils.data import Dataset
from csv import reader
import torch

class EHRDataset(Dataset):
    def __init__(self, dataframe, max_len=None, tokenizer=None):
        super(EHRDataset, self).__init__()
        
        self.age = dataframe['age']#.split(',')
        self.code = dataframe['icd_code']#.split(',')       
        self.max_len = max_len
        
        self.tokenizer = tokenizer
        
        
    def __getitem__(self, index):
        
        
        age = self.age[index].replace('SEP', '[SEP]').split(',')
        codes = self.code[index].replace('SEP', '[SEP]').split(',')
        
        # Start by tokenizing the visits
        # Then add special tokens
        # Transform the data, when index is in into:
        
        age.insert(0, '[CLS]')
        codes.insert(0, '[CLS]')
        age_ids = self.tokenizer.convert_tokens_to_ids(age)
        code_ids = self.tokenizer.convert_tokens_to_ids(codes)
        
        def transform():
            
            patient = {'age':[], 'codes':[]}
            ages_ = []
            codes_ = []
            for ag, code in zip(age, codes): 
                if ag != 'SEP':
                    ages_.append(int(ag)), codes_.append(code)
                else:
                    patient['age'].append(ages_), patient['codes'].append(codes_)
                    ages_, codes_ = [], []
                    
            return patient
                    
                    
        
        
       
        
        return torch.LongTensor(age_ids), torch.LongTensor(code_ids)
        
    def __len__(self):
        return len(self.code)
        
        
        
        
        
    
    
