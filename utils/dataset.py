
import sys
sys.path.insert(1, '../')

from torch.utils.data import Dataset
from csv import reader
import torch
import random
import pickle
import os
import numpy as np


from utils.functions import *

## Dataset used for pretraining with MLM
class EHRDataset(Dataset):
    def __init__(self, dataframe, conditional_files, save_folder, feature_types, max_len=64, tokenizer=None, run_type='train'): # Should be changed 
        super(EHRDataset, self).__init__()
        
        # Get conditional files
        dd_path = conditional_files['dd'] # diagnose-diagnose
        dp_path = conditional_files['dp'] # diagnose-procedure
        dm_path = conditional_files['dm'] # diagnose-medication
        
        pp_path = conditional_files['pp'] # procedure-procedure
        pd_path = conditional_files['pd'] # procedure-diagnose
        pm_path = conditional_files['pm'] # procedure-medications
        
        mm_path = conditional_files['mm'] # medication-medication
        md_path = conditional_files['md'] # medication-diagnose
        mp_path = conditional_files['mp'] # medication-procedure
        
        # Load conditional files
        dd = pickle.load(open(dd_path, 'rb'))
        dp = pickle.load(open(dp_path, 'rb'))
        dm = pickle.load(open(dm_path, 'rb'))
        
        pp = pickle.load(open(pp_path, 'rb'))
        pd = pickle.load(open(pd_path, 'rb'))
        pm = pickle.load(open(pm_path, 'rb'))
        
        mm = pickle.load(open(mm_path, 'rb'))
        mp = pickle.load(open(mp_path, 'rb'))
        md = pickle.load(open(md_path, 'rb'))
        
        self.codemaps = (dd, dp, dm, pp, pd, pm, mm, md, mp)
        
        self.dataframe = dataframe
        self.max_len = max_len
        self.ids = dataframe.subject_id
        self.tokenizer = tokenizer
        self.run_type = run_type
        self.save_folder = save_folder
        
        self.use_d = feature_types['diagnosis'] # Use diagnosis
        self.use_m = feature_types['medications'] # Use medications
        self.use_p = feature_types['procedures'] # Use procedures
        
        def save_dataset(patient_dict):
            path = self.save_folder + '_MLM_' + self.run_type
            save_file = open(path + '.pkl', 'wb')
            pickle.dump(patient_dict, save_file)
            save_file.close()
        
        def load_dataset():
            path = self.save_folder + '_MLM_' + self.run_type
            dataset_file = open(path + '.pkl', 'rb')
            dataset = pickle.load(dataset_file)
            dataset_file.close()
            
            return dataset
            
            
        def _transform_data(data):
            
            """
            Transforms the records from containing arrays into a dictionary: 
            patient_record = {patientid1: records, patientid2: records, ...}
         
            """
            
            patient_records = {}
            
            ## Implement the option of selecting what features to include into the model 
            for _, row in data.iterrows():
                patient_id, nradm, icd_codes, ndc_codes = row['subject_id'], len(list(row['hadm_id'])), list(row['diagnos_code']), list(row['medication_code'])
                procedure_codes = list(row['procedure_code'])
                age, gender = list(row['age']), list(row['gender'])
                admission_input, adm_age, adm_gender = [], [], []
                
                for i in range(nradm):
                    tot_len = 1 # 1 For SEP token
                    proc_codes = procedure_codes[i]
                    if self.use_d:
                        admission_input.extend(icd_codes[i])
                        tot_len += len(icd_codes[i])
                        
                    if self.use_m:
                        admission_input.extend(ndc_codes[i])
                        tot_len += len(ndc_codes[i])
                    
                    if ((self.use_p) and (procedure_codes[i] != -1)):
                        admission_input.extend(procedure_codes[i])
                        tot_len += len(procedure_codes[i])
                        
                    admission_input.extend(['[SEP]'])
                    adm_age.extend([str(int(age[i]))]*tot_len)
                    adm_gender.extend(gender*tot_len)
                                   
                patient_records[patient_id] = [admission_input, adm_age, adm_gender]
                
            return patient_records
        
        if os.path.isfile(self.save_folder + '_MLM_' + self.run_type + '.pkl'):
            print("Loading data")
            records = load_dataset()
        else:
            print("Transforming data")
            records = _transform_data(dataframe)
            print("Saving data")
            save_dataset(records)
            
        self.patdata = records
        
        """
        The final representation after transforming should be:
        
        records[patientid1] -> [admissioninput (tokens), age, gender], where admissioninput will look like:
        
        admissioninput -> [code1, code2, medication1, medication2, tobacco, alcohol use, '[SEP]', ...]
        
        """
        
    def __getitem__(self, index):
    
        patid = self.ids.iloc[index]
        
        record = self.patdata[patid]
                
        codes = record[0] # inputtokens
        
        age = record[1] # ages over the admissions
        
        gender = record[2]
        
        ## Cut sequence to the end of a visit
        codes = codes[(-self.max_len + 1):]
        age = age[(-self.max_len + 1):]
        gender = gender[(-self.max_len + 1):]
        
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
        
        prior_guide = create_prior_guide(self.codemaps, codes)
        prior_guide = np.array(prior_guide).reshape(self.max_len, -1)
        
        age_ids = self.tokenizer.convert_tokens_to_ids(age, 'age') # Convert age to ids
        gender_ids = self.tokenizer.convert_tokens_to_ids(gender, 'gender') # Convert gender to ids
        code_ids = self.tokenizer.convert_tokens_to_ids(codes, 'code') # Convert codes to ids
        label = self.tokenizer.convert_tokens_to_ids(label, 'code') # Convert labels to ids    
                
        return torch.LongTensor(age_ids), torch.LongTensor(gender_ids), torch.LongTensor(code_ids), torch.LongTensor(position), torch.LongTensor(segment), torch.LongTensor(mask), torch.LongTensor(label), torch.FloatTensor(prior_guide)
          
    def __len__(self):
        return len(self.dataframe)
    
    
    
class EHRDatasetReadmission(Dataset):
    
    def __init__(self, dataframe, conditional_files, save_folder, feature_types, max_len=64, tokenizer=None, run_type='train', nvisits=2):
        
       # Get conditional files
        dd_path = conditional_files['dd'] # diagnose-diagnose
        dp_path = conditional_files['dp'] # diagnose-procedure
        dm_path = conditional_files['dm'] # diagnose-medication
        
        pp_path = conditional_files['pp'] # procedure-procedure
        pd_path = conditional_files['pd'] # procedure-diagnose
        pm_path = conditional_files['pm'] # procedure-medications
        
        mm_path = conditional_files['mm'] # medication-medication
        md_path = conditional_files['md'] # medication-diagnose
        mp_path = conditional_files['mp'] # medication-procedure
        
        # Load conditional files
        dd = pickle.load(open(dd_path, 'rb'))
        dp = pickle.load(open(dp_path, 'rb'))
        dm = pickle.load(open(dm_path, 'rb'))
        
        pp = pickle.load(open(pp_path, 'rb'))
        pd = pickle.load(open(pd_path, 'rb'))
        pm = pickle.load(open(pm_path, 'rb'))
        
        mm = pickle.load(open(mm_path, 'rb'))
        mp = pickle.load(open(mp_path, 'rb'))
        md = pickle.load(open(md_path, 'rb'))
        
        self.codemaps = (dd, dp, dm, pp, pd, pm, mm, md, mp)
        
        self.use_d = feature_types['diagnosis'] # Use diagnosis
        self.use_m = feature_types['medications'] # Use medications
        self.use_p = feature_types['procedures'] # Use procedures
        
        #data = dataframe[dataframe['hadm_id'].map(len) >= nvisits]
        
        self.data = dataframe
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.ids = data.subject_id
        self.nvisits = nvisits
        self.run_type = run_type
        self.save_folder = save_folder
        
        def save_dataset(patient_dict):
            path = self.save_folder + '_readmission_' + self.run_type
            save_file = open(path + '.pkl', 'wb')
            pickle.dump(patient_dict, save_file)
            save_file.close()
        
        def load_dataset():
            path = self.save_folder + '_readmission_' + self.run_type
            dataset_file = open(path + '.pkl', 'rb')
            dataset = pickle.load(dataset_file)
            dataset_file.close()
            
            return dataset
        
        def _transform_data(data):
            patient_records = {} 
            for _, row in data.iterrows():
                patient_id, nradm, icd_codes, ndc_codes = row['subject_id'], len(list(row['hadm_id'])), list(row['diagnos_code']), list(row['medication_code'])
                procedure_codes = list(row['procedure_code'])
                age, gender = list(row['age']), list(row['gender'])
                lab = list(row['label'])
                admission_input, adm_age, adm_gender, labels = [], [], [], []
                
                nvisits = self.nvisits
                for i in range(nvisits):
                    tot_len = 1 # 1 For SEP token
                    
                    if self.use_d:
                        admission_input.extend(icd_codes[i])
                        tot_len += len(icd_codes[i])
                        
                    if self.use_m:
                        admission_input.extend(ndc_codes[i])
                        tot_len += len(ndc_codes[i])
                    
                    if ((self.use_p) and (procedure_codes[i] != -1)):
                        admission_input.extend(procedure_codes[i])
                        tot_len += len(procedure_codes[i])
                        
                    admission_input.extend(['[SEP]'])
                    adm_age.extend([str(int(age[i]))]*tot_len)
                    adm_gender.extend(gender*tot_len)
                    
                labels = [int(lab[nvisits - 1])]
                    
                patient_records[patient_id] = [admission_input, adm_age, adm_gender, labels]
                
            return patient_records
        
        if os.path.isfile(self.save_folder + '_readmission_' + self.run_type + '.pkl'):
            print("Loading data")
            records = load_dataset()
        else:
            print("Transforming data")
            records = _transform_data(self.data)
            print("Saving data")
            save_dataset(records)
            
        self.patdata = records
        
    def __getitem__(self, index):
        
        patid = self.ids.iloc[index]
        record = self.patdata[patid]
                
        codes = record[0] # inputtokens
        
        
        age = record[1] # ages over the admissions
        
        gender = record[2]
        
        labels = record[3]
        code_mapping = self.codemaps
        
        ## Cut sequence to the end of a visit
        codes = codes[(-self.max_len + 1):]       
        
        age = age[(-self.max_len + 1):]
        gender = gender[(-self.max_len + 1):]
        
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
        
        prior_guide = create_prior_guide(self.codemaps, codes)
        prior_guide = np.array(prior_guide).reshape(self.max_len, -1)
        
        age_ids = self.tokenizer.convert_tokens_to_ids(age, 'age')
        gender_ids = self.tokenizer.convert_tokens_to_ids(gender, 'gender')
        code_ids = self.tokenizer.convert_tokens_to_ids(codes, 'code')
            
        return torch.LongTensor(age_ids), torch.LongTensor(gender_ids), torch.LongTensor(code_ids), torch.LongTensor(position), torch.LongTensor(segment), torch.LongTensor(mask), torch.LongTensor(label), torch.FloatTensor(prior_guide)
        
    def __len__(self):
        return len(self.data)
    
    
    
## This dataset is used for the prediction tasks

class EHRDatasetCodePrediction(Dataset):
    
    def __init__(self, dataframe, conditional_files, save_folder, feature_types, max_len=64, tokenizer=None, run_type='train'):
        
        
        # Get conditional files
        dd_path = conditional_files['dd'] # diagnose-diagnose
        dp_path = conditional_files['dp'] # diagnose-procedure
        dm_path = conditional_files['dm'] # diagnose-medication
        
        pp_path = conditional_files['pp'] # procedure-procedure
        pd_path = conditional_files['pd'] # procedure-diagnose
        pm_path = conditional_files['pm'] # procedure-medications
        
        mm_path = conditional_files['mm'] # medication-medication
        md_path = conditional_files['md'] # medication-diagnose
        mp_path = conditional_files['mp'] # medication-procedure
        
        # Load conditional files
        dd = pickle.load(open(dd_path, 'rb'))
        dp = pickle.load(open(dp_path, 'rb'))
        dm = pickle.load(open(dm_path, 'rb'))
        
        pp = pickle.load(open(pp_path, 'rb'))
        pd = pickle.load(open(pd_path, 'rb'))
        pm = pickle.load(open(pm_path, 'rb'))
        
        mm = pickle.load(open(mm_path, 'rb'))
        mp = pickle.load(open(mp_path, 'rb'))
        md = pickle.load(open(md_path, 'rb'))
        
        self.use_d = feature_types['diagnosis'] # Use diagnosis
        self.use_m = feature_types['medications'] # Use medications
        self.use_p = feature_types['procedures'] # Use procedures
        
        self.codemaps = (dd, dp, dm, pp, pd, pm, mm, md, mp)
            
        self.data = dataframe[dataframe['hadm_id'].str.len() >= 2]
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.ids = self.data.subject_id
        self.run_type = run_type
        self.save_folder = save_folder
        
        
        def save_dataset(patient_dict):
            path = self.save_folder + '_nextvisit_' + self.run_type
            save_file = open(path + '.pkl', 'wb')
            pickle.dump(patient_dict, save_file)
            save_file.close()
        
        def load_dataset():
            path = self.save_folder + '_nextvisit_' + self.run_type
            dataset_file = open(path + '.pkl', 'rb')
            dataset = pickle.load(dataset_file)
            dataset_file.close()
            
            return dataset
        
        def _transform_data(data):
            patient_records = {} 
            for _, row in data.iterrows():
                patient_id, nradm, icd_codes, ndc_codes = row['subject_id'], len(list(row['hadm_id'])), list(row['diagnos_code']), list(row['medication_code'])
                procedure_codes = list(row['procedure_code'])
                age, gender = list(row['age']), list(row['gender'])
                admission_input, adm_age, adm_gender, labels = [], [], [], []
                #NextVisit = self.NextVisit
                    
                for i in range(nradm - 1):
                    tot_len = 1
                    if self.use_d:
                        admission_input.extend(icd_codes[i])
                        tot_len += len(icd_codes[i])
                        
                    if self.use_m:
                        admission_input.extend(ndc_codes[i])
                        tot_len += len(ndc_codes[i])
                    
                    if ((self.use_p) and (procedure_codes[i] != -1)):
                        admission_input.extend(procedure_codes[i])
                        tot_len += len(procedure_codes[i])
                        
                    admission_input.extend(['[SEP]'])
                    adm_age.extend([str(int(age[i]))]*tot_len)
                    adm_gender.extend(gender*tot_len)
                    
                labels = icd_codes[nradm - 1]
                patient_records[patient_id] = [admission_input, adm_age, adm_gender, labels]
                
            return patient_records
        
        #path = '../data/pytorch_datasets/CodePrediction_' + self.run_type + '_' + str(self.NextVisit) + '.pkl'
        if os.path.isfile(self.save_folder + '_nextvisit_' + self.run_type + '.pkl'):
            print("Loading data")
            records = load_dataset()
        else:
            print("Transforming data")
            records = _transform_data(self.data)
            print("Saving data")
            save_dataset(records)
            
        self.patdata = records
        
    def __getitem__(self, index):
        
        patid = self.ids.iloc[index]
        record = self.patdata[patid]
                
        codes = record[0] # inputtokens
        age = record[1] # ages over the admissions
        gender = record[2]
        labels = record[3]
        code_mapping = self.codemaps
        
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
        
        prior_guide = create_prior_guide(self.codemaps, codes)
        prior_guide = np.array(prior_guide).reshape(self.max_len, -1)
        
        age_ids = self.tokenizer.convert_tokens_to_ids(age, 'age')
        gender_ids = self.tokenizer.convert_tokens_to_ids(gender, 'gender')
        code_ids = self.tokenizer.convert_tokens_to_ids(codes, 'code')
        
        labels = self.tokenizer.convert_tokens_to_ids(labels, 'label')
            
        return torch.LongTensor(age_ids), torch.LongTensor(gender_ids), torch.LongTensor(code_ids), torch.LongTensor(position), torch.LongTensor(segment), torch.LongTensor(mask), torch.LongTensor(labels), torch.FloatTensor(prior_guide)
        
    def __len__(self):
        return len(self.data)
        
        
    
    
