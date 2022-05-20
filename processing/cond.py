"""Copyright 2019 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import division
from __future__ import print_function

#import cPickle as pickle
import _pickle as pickle
import csv
import os
import sys
import numpy as np
import sklearn.model_selection as ms
import tensorflow as tf
import pandas as pd



class Patient(object):
    def __init__(self, patient_id):
        self.patient_id = patient_id
        self.visits = []
        self.prior_indicies = []
        self.prior_values = []
        
    def add_visit(self, visit_id, indicies, values):
        self.visits.append(visit_id)
        self.prior_indicies.append(indicies)
        self.prior_values.append(values)
    
    def get_information(self):
        #print("TYPE: {}".format(type(self.prior_values[0])))
        
        infdict = {'subject_id':self.patient_id, 'prior_values': self.prior_values, 'prior_indicies': self.prior_indicies} 
        return infdict
    

class EncounterInfo(object):#en visit
    
    def __init__(self, patient_id, encounter_id, visit_number, readmission, icd_code, ndc, procedures):
        self.patient_id = str(patient_id)
        self.encounter_id = str(encounter_id)
        self.readmission = str(readmission)
        self.rx_ids = []
        self.dx_ids = icd_code
        self.labs = {}
        self.physicals = []
        self.treatments = ndc
        self.procedures = procedures
       # self.bmi = bmi
        
        self.visit_number = visit_number
    def __str__(self):
        return ("patient_id:"+str(self.patient_id)+" visit number:"+str(self.visit_number)+" hadm_id:"+str(self.encounter_id)+"readmission:"+str(self.readmission))
    

def process_patient(infile, encounter_dict, hour_threshold=24):
    
    data = pd.read_parquet(infile)    
    
    count = 0
    r_count = 0
    nr_count = 0
    patient_dict = {}
    for rowindex, line in data.iterrows():
        if count % 10000 == 0:
            sys.stdout.write('%d\r' % count)
            sys.stdout.flush()
        patient_id = line['subject_id']
        encounter_id = line['hadm_id']
        readmission = line["label"]
        icd_code = line["diagnos_code"]
        ndc = line["medication_code"]
        procedures = line["procedure_code"]
        bmi = line["bmi_value"]

        if patient_id not in patient_dict:
            patient_dict[patient_id] = []
        #patient_dict[patient_id].append((encounter_timestamp, encounter_id))
        patient_dict[patient_id].append(encounter_id)
        patient_dict[patient_id].append(readmission)
        patient_dict[patient_id].append(icd_code)
        patient_dict[patient_id].append(ndc)
        patient_dict[patient_id].append(procedures)
        patient_dict[patient_id].append(bmi)

        count+=1
  
    enc_readmission_dict = {}
    for patient_id, information in patient_dict.items():
        if sum(information[1]) > 0:
            for visit_no in range(information[0].shape[0]):
                c_hadm = information[0][visit_no]
                c_readmission = information[1][visit_no]
                c_icd_code = information[2][visit_no]
                c_ndc = information[3][visit_no]
                c_procedures = information[4][visit_no]
                #c_bmi = information[5][visit_no]
                
                
                if c_readmission == 1:
                    r_count += 1
                else:
                    nr_count += 1
                ei = EncounterInfo(patient_id, c_hadm, visit_no, c_readmission, c_icd_code, c_ndc, c_procedures)
                if information[0][visit_no] in encounter_dict:
                    print('Duplicate encounter ID!!')
                encounter_dict[c_hadm] = ei
  
    print("len after sampling",len(encounter_dict))
    print("total readmissions", r_count)
    print("total no readmissions", nr_count)
    del data
    return encounter_dict
    
    
def build_seqex(enc_dict,skip_duplicate=False,min_num_codes=1,max_num_codes=50):
    
    key_list = []
    seqex_list = []
    dx_str2int = {}
    treat_str2int = {}
    procedure_str2int = {}
    bmi_str2int = {}
    
    num_cut = 0
    num_duplicate = 0
    count = 0
    num_dx_ids = 0
    num_treatments = 0
    num_unique_dx_ids = 0
    num_unique_treatments = 0
    min_dx_cut = 0
    min_treatment_cut = 0
    max_dx_cut = 0
    max_treatment_cut = 0
    num_expired = 0
    num_readmission = 0
    
    print("len_enc",len(enc_dict))
    for _, enc in enc_dict.items():
        if skip_duplicate:
            if (len(enc.dx_ids) > len(set(enc.dx_ids)) or
                len(enc.treatments) > len(set(enc.treatments))):
                num_duplicate += 1
                continue
        '''
        if len(set(enc.dx_ids)) < min_num_codes:
            min_dx_cut += 1
            continue

        if len(set(enc.treatments)) < min_num_codes:
            min_treatment_cut += 1
            continue

        if len(set(enc.dx_ids)) > max_num_codes:
            max_dx_cut += 1
            continue

        if len(set(enc.treatments)) > max_num_codes:
            max_treatment_cut += 1
            continue
        '''
        
        count += 1
        num_dx_ids += len(enc.dx_ids)
        num_treatments += len(enc.treatments)
        num_unique_dx_ids += len(set(enc.dx_ids))
        num_unique_treatments += len(set(enc.treatments))
        
        for dx_id in enc.dx_ids:
            if dx_id not in dx_str2int:
                dx_str2int[dx_id] = len(dx_str2int)

        for treat_id in enc.treatments:
            if treat_id not in treat_str2int:
                treat_str2int[treat_id] = len(treat_str2int)
        
        for procedure_id in enc.procedures:
            if procedure_id not in procedure_str2int:
                procedure_str2int[procedure_id] = len(procedure_str2int)
        
        #    if bmi_id not in bmi_str2_int:
        #        bmi_str2int[bmi_id] = len(bmi_str2int)
                
                
        seqex = tf.train.SequenceExample()
        string = (enc.patient_id + ':' + enc.encounter_id).encode()
        seqex.context.feature['patientId'].bytes_list.value.append(string)
    
        if enc.readmission == True:
            seqex.context.feature['label.readmission'].int64_list.value.append(1)
            num_readmission += 1
        else:
            seqex.context.feature['label.readmission'].int64_list.value.append(0)
        
        # Add diagnoses
        dx_ids = seqex.feature_lists.feature_list['dx_ids'] 
        enc.dx_ids = [str(x).encode() for x in enc.dx_ids]
        dx_ids.feature.add().bytes_list.value.extend(list(set(enc.dx_ids)))

        dx_int_list = [dx_str2int[int(item.decode())] for item in list(set(enc.dx_ids))]
        dx_ints = seqex.feature_lists.feature_list['dx_ints']
        dx_ints.feature.add().int64_list.value.extend(dx_int_list)
        
        # Add medications
        enc.treatments = [str(x).encode() for x in enc.treatments]
        med_ids = seqex.feature_lists.feature_list['med_ids']
        med_ids.feature.add().bytes_list.value.extend(list(set(enc.treatments)))

        med_int_list = [treat_str2int[int(item.decode())] for item in list(set(enc.treatments))]
        med_ints = seqex.feature_lists.feature_list['med_ints']
        med_ints.feature.add().int64_list.value.extend(med_int_list)
        
        # Add procedures
        enc.procedures = [str(x).encode() for x in enc.procedures]
        proc_ids = seqex.feature_lists.feature_list['proc_ids']
        proc_ids.feature.add().bytes_list.value.extend(list(set(enc.procedures)))

        proc_int_list = [procedure_str2int[int(item.decode())] for item in list(set(enc.procedures))]
        proc_ints = seqex.feature_lists.feature_list['proc_ints']
        proc_ints.feature.add().int64_list.value.extend(proc_int_list)
        
        # Add bmi
       # enc.bmi = [str(x).encode() for x in enc.bmi]
       # bmi_ids = seqex.feature_lists.feature_list['bmi_ids']
       # bmi_ids.feature.add().bytes_list.value.extend(list(set(enc.bmi)))

       # bmi_int_list = [treat_str2int[int(item.decode())] for item in list(set(enc.bmi))]
       # bmi_ints = seqex.feature_lists.feature_list['bmi_ints']
       # bmi_ints.feature.add().int64_list.value.extend(bmi_int_list)        

        seqex_list.append(seqex)
        key = seqex.context.feature['patientId'].bytes_list.value[0]
        key_list.append(key)

    print('Filtered encounters due to duplicate codes: %d' % num_duplicate)
    print('Filtered encounters due to thresholding: %d' % num_cut)
    print('Average num_dx_ids: %f' % (num_dx_ids / count))
    print('Average num_treatments: %f' % (num_treatments / count))
    print('Average num_unique_dx_ids: %f' % (num_unique_dx_ids / count))
    print('Average num_unique_treatments: %f' % (num_unique_treatments / count))
    print('Min dx cut: %d' % min_dx_cut)
    print('Min treatment cut: %d' % min_treatment_cut)
    print('Max dx cut: %d' % max_dx_cut)
    print('Max treatment cut: %d' % max_treatment_cut)
    print('Number of expired: %d' % num_expired)
    print('Number of readmission: %d' % num_readmission)

    return key_list, seqex_list, dx_str2int, treat_str2int


def count_conditional_prob_dp(seqex_list, output_path, train_key_set=None):
    
  
    print("Conditional probabilites")
    dx_freqs = {} # Diagnoses frequency
    med_freqs = {} # Medication freqs
    proc_freqs = {} # Procedure freqs
    
    dd_freqs = {} # Diagnose-Diagnose freqs
    pp_freqs = {} # Procedure-Procedure freqs
    mm_freqs = {} # Medication-Medication freqs
    dp_freqs = {} # Diagnose-procedure freqs 
    dm_freqs = {} # Diagnose-Medication freqs
    pm_freqs = {} # Procedure-Medication freqs
    
    
    total_visit = 0
    for seqex in seqex_list:
        if total_visit % 1000 == 0:
            sys.stdout.write('Visit count: %d\r' % total_visit)
            sys.stdout.flush()

        key = seqex.context.feature['patientId'].bytes_list.value[0]
        if (train_key_set is not None and key not in train_key_set):
            total_visit += 1
            continue

        dx_ids = seqex.feature_lists.feature_list['dx_ids'].feature[0].bytes_list.value # Diagnoses
        med_ids = seqex.feature_lists.feature_list['med_ids'].feature[0].bytes_list.value # Meds
        proc_ids = seqex.feature_lists.feature_list['proc_ids'].feature[0].bytes_list.value # Procedures
        
        # Diagnoses P(D)
        for dx in dx_ids:
            dx = dx.decode()
            if dx not in dx_freqs:
                dx_freqs[dx] = 0
            dx_freqs[dx] += 1
            
        # Medications P(M)
        for med in med_ids:
            med = med.decode()
            if med not in med_freqs:
                med_freqs[med] = 0
            med_freqs[med] += 1
        
        # Procedures P(Proc)
        for proc in proc_ids:
            proc = proc.decode()
            if proc == '-1':
                continue
                
            if proc not in proc_freqs:
                proc_freqs[proc] = 0
            proc_freqs[proc] += 1
            
        # Diagnoses and Medications P(D and M)
        for dx in dx_ids:
            for med in med_ids:
                dm = dx.decode() + ',' + med.decode()
                if dm not in dm_freqs:
                    dm_freqs[dm] = 0
                dm_freqs[dm] += 1    
        
        # Diagnoses and procedures P(D and Proc)
        for dx in dx_ids:
            for proc in proc_ids:
                if proc.decode() == '-1':
                    continue
                dp = dx.decode() + ',' + proc.decode()
                if dp not in dp_freqs:
                    dp_freqs[dp] = 0
                dp_freqs[dp] += 1
                
        # Procedures and Medications P(Proc and Medications)
        for med in med_ids:
            for proc in proc_ids:
                if proc.decode() == '-1':
                    continue
                mp = med.decode() + ',' + proc.decode()
                if mp not in pm_freqs:
                    pm_freqs[mp] = 0
                pm_freqs[mp] += 1
                
        # Diagnoses and Diagnoses P(D and D)
        for dx1 in dx_ids:
            for dx2 in dx_ids:
                dd = dx1.decode() + ',' + dx2.decode()
                if dd not in dd_freqs:
                    dd_freqs[dd] = 0
                dd_freqs[dd] += 1
                
        # Procedures and Procedures P(Proc and Proc)
        for proc1 in proc_ids:
            if proc1.decode() == '-1':
                continue
            for proc2 in proc_ids:
                if proc2.decode() == '-1':
                    continue
                pp = proc1.decode() + ',' + proc2.decode()
                if pp not in pp_freqs:
                    pp_freqs[pp] = 0
                pp_freqs[pp] += 1
                
        # Procedures and Procedures P(M and M)
        for med1 in med_ids:
            for med2 in med_ids:
                mm = med1.decode() + ',' + med2.decode()
                if mm not in mm_freqs:
                    mm_freqs[mm] = 0
                mm_freqs[mm] += 1

        total_visit += 1

    dx_probs = dict([(k, v / float(total_visit)) for k, v in dx_freqs.items()]) # P(D)
    med_probs = dict([(k, v / float(total_visit)) for k, v in med_freqs.items()]) # P(M)
    proc_probs = dict([(k, v / float(total_visit)) for k, v in proc_freqs.items()]) # P(Procs)
    
    dp_probs = dict([(k, v / float(total_visit)) for k, v in dp_freqs.items()]) # P(D and Procs)
    dm_probs = dict([(k, v / float(total_visit)) for k, v in dm_freqs.items()]) # P(D and M)
    mp_probs = dict([(k, v / float(total_visit)) for k, v in pm_freqs.items()]) # P (M and P)
    
    dd_probs = dict([(k, v / float(total_visit)) for k, v in dd_freqs.items()]) # P(D and D)
    pp_probs = dict([(k, v / float(total_visit)) for k, v in pp_freqs.items()]) # P(Procs and Procs) 
    mm_probs = dict([(k, v / float(total_visit)) for k, v in mm_freqs.items()]) # P(M and M) 

    # Calculate dd, pp, and mm cond probs.   
    dd_cond_probs = {} # P(D|D)
    pp_cond_probs = {} # P(P|P)
    mm_cond_probs = {} # P(M|M)
    
    # P(D|D)
    for dx1, dx_prob1 in dx_probs.items():
        for dx2, dx_prob2 in dx_probs.items(): #P(A|B) = P (A and B) / P(B)
            dd = dx1 + ',' + dx2
            if dd in dd_probs:
                dd_cond_probs[dd] = dd_probs[dd] / dx_prob2
            else:
                dd_cond_probs[dd] = 0.0

    # P(P|P)
    for proc1, proc_prob1 in proc_probs.items():
        for proc2, proc_prob2 in proc_probs.items():
            pp = proc1 + ',' + proc2
            if pp in pp_probs:
                pp_cond_probs[pp] = pp_probs[pp] / proc_prob2
            else:
                pp_cond_probs[dd] = 0.0

    # P(M|M)
    for med1, med_prob1 in med_probs.items():
        for med2, med_prob2 in med_probs.items():
            mm = med1 + ',' + med2
            if mm in mm_probs:
                mm_cond_probs[mm] = mm_probs[mm] / med_prob2
            else:
                mm_cond_probs[mm] = 0.0
    
    
    dp_cond_probs = {} # P(D|P)
    dm_cond_probs = {} # P(D|M)
    
    pd_cond_probs = {} # P(P|D)
    pm_cond_probs = {} # P(P|M)
    
    md_cond_probs = {} # P(M|D)
    mp_cond_probs = {} # P(M|P)
    
    # P(D|P) and P(P|D)
    for dx, dx_prob in dx_probs.items():
        for proc, proc_prob in proc_probs.items():
            dp = dx + ',' + proc
            pd = proc + ',' + dx
            if dp in dp_probs:
                dp_cond_probs[dp] = dp_probs[dp] / proc_prob # P(D|P) = P(D and P) / P(P)
                pd_cond_probs[pd] = dp_probs[dp] / dx_prob # P(P|D) P (P and D) / P(D)
            else:
                dp_cond_probs[dp] = 0.0
                pd_cond_probs[pd] = 0.0
    
    # P(D|M) and P(M|D)
    for dx, dx_prob in dx_probs.items():
        for med, med_prob in med_probs.items():
            dm = dx + ',' + med
            md = med + ',' + dx
            if dm in dm_probs:
                dm_cond_probs[dm] = dm_probs[dm] / med_prob # P(D|M) = P(D and M) / P(M)
                md_cond_probs[md] = dm_probs[dm] / dx_prob # P(M|D) = P(M and D) / P(D)
            else:
                dm_cond_probs[dm] = 0.0
                md_cond_probs[md] = 0.0    
                
    # P(P|M) and P(M|P)
    for proc, proc_prob in proc_probs.items():
        for med, med_prob in med_probs.items():
            pm = proc + ',' + med
            mp = med + ',' + proc
            if mp in mp_probs:
                pm_cond_probs[pm] = mp_probs[mp] / med_prob # P(P|M) = P(P and M) / P(M)
                mp_cond_probs[mp] = mp_probs[mp] / proc_prob # P(M|P) = P(M and P) / P(P)
            else:
                pm_cond_probs[pm] = 0.0
                mp_cond_probs[mp] = 0.0 
    
    

    #pickle.dump(dx_probs, open(output_path + '/dx_probs.empirical.p', 'wb'), -1)
    #pickle.dump(proc_probs, open(output_path + '/proc_probs.empirical.p', 'wb'),-1)
    #pickle.dump(dp_probs, open(output_path + '/dp_probs.empirical.p', 'wb'), -1)

    pickle.dump(dp_cond_probs,open(output_path + '/dp_cond_probs.empirical.p', 'wb'), -1) # D-P
    pickle.dump(dm_cond_probs,open(output_path + '/dm_cond_probs.empirical.p', 'wb'), -1) # D-M
    pickle.dump(dd_cond_probs,open(output_path + '/dd_cond_probs.empirical.p', 'wb'), -1) # D-D
    
    pickle.dump(pd_cond_probs,open(output_path + '/pd_cond_probs.empirical.p', 'wb'), -1) # P-D
    pickle.dump(pm_cond_probs,open(output_path + '/pm_cond_probs.empirical.p', 'wb'), -1) # P-M
    pickle.dump(pp_cond_probs,open(output_path + '/pp_cond_probs.empirical.p', 'wb'), -1) # P-P
    
    pickle.dump(md_cond_probs,open(output_path + '/md_cond_probs.empirical.p', 'wb'), -1) # M-D
    pickle.dump(mp_cond_probs,open(output_path + '/mp_cond_probs.empirical.p', 'wb'), -1) # M-P
    pickle.dump(mm_cond_probs,open(output_path + '/mm_cond_probs.empirical.p', 'wb'), -1) # M-M
    
def add_sparse_prior_guide_dp(seqex_list,stats_path,key_set=None,max_num_codes=50):
    
    print('Loading conditional probabilities.')
    
    # dp and pd cond probs
    
    dp_cond_probs = pickle.load(
        open(stats_path + '/dp_cond_probs.empirical.p', 'rb'))
    pd_cond_probs = pickle.load(
        open(stats_path + '/pd_cond_probs.empirical.p', 'rb'))

    
  # dd and pp cond probs
    dd_cond_probs = pickle.load(
        open(stats_path + '/dd_cond_probs.empirical.p', 'rb'))
    
    pp_cond_probs = pickle.load(
        open(stats_path + '/pp_cond_probs.empirical.p', 'rb'))

    print('Adding prior guide.')
    total_visit = 0
    new_seqex_list = []
    for seqex in seqex_list:
        if total_visit % 1000 == 0:
            sys.stdout.write('Visit count: %d\r' % total_visit)
            sys.stdout.flush()

        key = seqex.context.feature['patientId'].bytes_list.value[0]
        if (key_set is not None and key not in key_set):
            total_visit += 1
            continue

        dx_ids = seqex.feature_lists.feature_list['dx_ids'].feature[
            0].bytes_list.value
        proc_ids = seqex.feature_lists.feature_list['proc_ids'].feature[
            0].bytes_list.value

        indices = []
        values = []
        for i, dx1 in enumerate(dx_ids):# New 
            dx1 = dx1.decode()
            for j, dx2 in enumerate(dx_ids):
                dx2 = dx2.decode()
                dd = dx1 + ',' + dx2
                indices.append(dd)
                #indices.append((i, max_num_codes + j))
                prob = 0.0 if dd not in dd_cond_probs else dd_cond_probs[dd]
                values.append(prob)
            
        for i, proc1 in enumerate(proc_ids):# New 
            proc1 = proc1.decode()
            for j, proc2 in enumerate(proc_ids):
                proc2 = proc2.decode()
                pp = proc1 + ',' + proc2
                indices.append(pp)
                #indices.append((i, max_num_codes + j))
                prob = 0.0 if pp not in pp_cond_probs else pp_cond_probs[pp]
                values.append(prob)        
                
        for i, dx in enumerate(dx_ids):
            dx = dx.decode()
            for j, proc in enumerate(proc_ids):
                proc = proc.decode()
                dp = dx + ',' + proc
                indices.append(dp)
        #indices.append((i, max_num_codes + j))
                prob = 0.0 if dp not in dp_cond_probs else dp_cond_probs[dp]
                values.append(prob)

        for i, proc in enumerate(proc_ids):
            proc = proc.decode()
            for j, dx in enumerate(dx_ids):
                dx = dx.decode()
                pd = proc + ',' + dx
                #indices.append(pd)
                indices.append(pd)
                prob = 0.0 if pd not in pd_cond_probs else pd_cond_probs[pd]
                values.append(prob)

        indices = list(np.array(indices).reshape([-1]))
        indices_feature = seqex.feature_lists.feature_list['prior_indices']
        indices = [ind.encode() for ind in indices]
        indices_feature.feature.add().bytes_list.value.extend(list(indices))
    
        values_feature = seqex.feature_lists.feature_list['prior_values']
        values_feature.feature.add().float_list.value.extend(values)

        new_seqex_list.append(seqex)
        total_visit += 1
        
    del seqex_list
    return new_seqex_list
    
    
def create_patient_objects(seqex_list, path_to_data):
    patients = {}
    print("create patient objects")
    seqx = 0
    for seqex in seqex_list:
        if seqx % 1000 == 0:
            sys.stdout.write('seqx count: %d\r' % seqx)
            sys.stdout.flush()
        
        pat_id, enc_id = str(seqex.context.feature['patientId'].bytes_list.value[0])[2:].split(':')
        indicies = [x.decode() for x in seqex.feature_lists.feature_list['prior_indices'].feature[0].bytes_list.value] 
        values = [x for x in seqex.feature_lists.feature_list['prior_values'].feature[0].float_list.value]
          
        if pat_id not in patients:
            patients[pat_id] = 0
            pat_temp = Patient(patient_id=pat_id)
            patients[pat_id] = pat_temp
        else:
            pat_temp = patients[pat_id]
            
        pat_temp.add_visit(enc_id, indicies, values)
        #patients[pat_id] = pat_temp
        
        seqx += 1
        del pat_id
        del enc_id
        del indicies
        del values
        
    del seqex_list
    print("")
    print("Saving to parquet")
    df = pd.DataFrame(columns=['subject_id', 'prior_values', 'prior_indicies'])
    patinfo = []
    for key, item in patients.items():
        info = item.get_information()
        patinfo.append(info)
    
    df = df.append(patinfo, ignore_index=True)
    df.to_parquet(path_to_data + '/prior_table_2')    
        
    
"""Set <input_path> to where the raw eICU CSV files are located.
Set <output_path> to where you want the output files to be.
"""
def main(argv):
    
    input_path = argv[1]
    output_path = argv[2]
    num_fold = 1

    patient_file = input_path
 
    encounter_dict = {}
    print('Processing data')
    encounter_dict = process_patient(patient_file, encounter_dict, hour_threshold=24)
    key_list, seqex_list, dx_map, proc_map = build_seqex(encounter_dict, skip_duplicate=False, min_num_codes=1, max_num_codes=50)
    
    del encounter_dict
    
   # pickle.dump(dx_map, open(output_path + '/dx_map.p', 'wb'), -1)
   # pickle.dump(proc_map, open(output_path + '/proc_map.p', 'wb'), -1)

    stats_path = output_path + '/test' #stats_path = output_path + '/train_stats'
    count_conditional_prob_dp(seqex_list, stats_path)
    #train_seqex = add_sparse_prior_guide_dp(seqex_list, stats_path, max_num_codes=50)
    #create_patient_objects(train_seqex, input_path) 

if __name__ == '__main__':
    main(sys.argv)
