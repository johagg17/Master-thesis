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
    
    def __init__(self, patient_id, encounter_id, visit_number, readmission, icd_code, ndc):
        self.patient_id = str(patient_id)
        self.encounter_id = str(encounter_id)
        self.readmission = str(readmission)
        self.rx_ids = []
        self.dx_ids = icd_code
        self.labs = {}
        self.physicals = []
        self.treatments = ndc
        self.visit_number = visit_number
    def __str__(self):
        return ("patient_id:"+str(self.patient_id)+" visit number:"+str(self.visit_number)+" hadm_id:"+str(self.encounter_id)+"readmission:"+str(self.readmission))

def process_patient(infile, encounter_dict, hour_threshold=24):
    
    
  data = pd.read_parquet(infile)
  data['label'] = data['label'].apply(lambda x: np.append(x, False))
    
    
  count = 0
  r_count = 0
  nr_count = 0
  patient_dict = {}
  for rowindex, line in data.iterrows():
    if count == 1000:
        break
    if count % 10000 == 0:
      sys.stdout.write('%d\r' % count)
      sys.stdout.flush()
    patient_id = line['subject_id']
    encounter_id = line['hadm_id']
    readmission = line["label"]
    icd_code = line["diagnos_code"]
    ndc = line["medication_code"]
    
    if patient_id not in patient_dict:
        patient_dict[patient_id] = []
    #patient_dict[patient_id].append((encounter_timestamp, encounter_id))
    patient_dict[patient_id].append(encounter_id)
    patient_dict[patient_id].append(readmission)
    patient_dict[patient_id].append(icd_code)
    patient_dict[patient_id].append(ndc)
    
    count+=1
  
  enc_readmission_dict = {}
  for patient_id, information in patient_dict.items():
    if sum(information[1]) > 0:
        for visit_no in range(information[0].shape[0]):
          #print(patient_id)
          c_hadm = information[0][visit_no]
          c_readmission = information[1][visit_no]
          c_icd_code = information[2][visit_no]
          c_ndc = information[3][visit_no]
          if c_readmission == 1:
            r_count += 1
          else:
            nr_count += 1
          ei = EncounterInfo(patient_id, c_hadm, visit_no, c_readmission, c_icd_code, c_ndc)
          if information[0][visit_no] in encounter_dict:
            print('Duplicate encounter ID!!')
          encounter_dict[c_hadm] = ei
  
  print("len after sampling",len(encounter_dict))
  print("total readmissions", r_count)
  print("total no readmissions", nr_count)
  return encounter_dict



def build_seqex(enc_dict,
                skip_duplicate=False,
                min_num_codes=1,
                max_num_codes=50):
  key_list = []
  seqex_list = []
  dx_str2int = {}
  treat_str2int = {}
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
  #yeet = 0
  #for index, line in enc_dict.iterrows():
  #  if line.readmission == "1":
  #    yeet += 1
    
  #print("readmissions again",yeet)
  print("len_enc",len(enc_dict))
  for _, enc in enc_dict.items():
    #print(enc)
    #sys.exit(0)
    if skip_duplicate:
      if (len(enc.dx_ids) > len(set(enc.dx_ids)) or
          len(enc.treatments) > len(set(enc.treatments))):
        num_duplicate += 1
        continue

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

    seqex = tf.train.SequenceExample()
    string = (enc.patient_id + ':' + enc.encounter_id).encode()
    seqex.context.feature['patientId'].bytes_list.value.append(string)
    
    if enc.readmission == True:
        seqex.context.feature['label.readmission'].int64_list.value.append(1)
        num_readmission += 1
    else:
        seqex.context.feature['label.readmission'].int64_list.value.append(0)
    
    dx_ids = seqex.feature_lists.feature_list['dx_ids']
    enc.dx_ids = [str(x).encode() for x in enc.dx_ids]
    dx_ids.feature.add().bytes_list.value.extend(list(set(enc.dx_ids)))
   
    dx_int_list = [dx_str2int[int(item.decode())] for item in list(set(enc.dx_ids))]
    dx_ints = seqex.feature_lists.feature_list['dx_ints']
    dx_ints.feature.add().int64_list.value.extend(dx_int_list)

    enc.treatments = [str(x).encode() for x in enc.treatments]
    proc_ids = seqex.feature_lists.feature_list['proc_ids']
    proc_ids.feature.add().bytes_list.value.extend(list(set(enc.treatments)))

    proc_int_list = [treat_str2int[int(item.decode())] for item in list(set(enc.treatments))]
    proc_ints = seqex.feature_lists.feature_list['proc_ints']
    proc_ints.feature.add().int64_list.value.extend(proc_int_list)

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


def select_train_valid_test(key_list, random_seed=1234):
  key_train, key_temp = ms.train_test_split(
      key_list, test_size=0.2, random_state=random_seed)
  key_valid, key_test = ms.train_test_split(
      key_temp, test_size=0.5, random_state=random_seed)
  return key_train, key_valid, key_test


def count_conditional_prob_dp(seqex_list, output_path, train_key_set=None):
  
  print("Conditional probabilites")
  dx_freqs = {}
  proc_freqs = {}
  dp_freqs = {}
  
  dd_freqs = {}
  pp_freqs = {}
    
  total_visit = 0
  for seqex in seqex_list:
    if total_visit % 1000 == 0:
      sys.stdout.write('Visit count: %d\r' % total_visit)
      sys.stdout.flush()

    key = seqex.context.feature['patientId'].bytes_list.value[0]
    if (train_key_set is not None and key not in train_key_set):
      total_visit += 1
      continue

    dx_ids = seqex.feature_lists.feature_list['dx_ids'].feature[
        0].bytes_list.value
    proc_ids = seqex.feature_lists.feature_list['proc_ids'].feature[
        0].bytes_list.value

    for dx in dx_ids:
      dx = dx.decode()
      if dx not in dx_freqs:
        dx_freqs[dx] = 0
      dx_freqs[dx] += 1

    for proc in proc_ids:
      proc = proc.decode()
      if proc not in proc_freqs:
        proc_freqs[proc] = 0
      proc_freqs[proc] += 1

    for dx in dx_ids:
      for proc in proc_ids:
        #print("Proc:, ", proc)
        dp = dx.decode() + ',' + proc.decode()
        if dp not in dp_freqs:
          dp_freqs[dp] = 0
        dp_freqs[dp] += 1
    
    for dx1 in dx_ids:
        for dx2 in dx_ids:
            dd = dx1.decode() + ',' + dx2.decode()
            if dd not in dd_freqs:
                dd_freqs[dd] = 0
            dd_freqs[dd] += 1
    
    for proc1 in proc_ids:
        for proc2 in proc_ids:
            pp = proc1.decode() + ',' + proc2.decode()
            if pp not in pp_freqs:
                pp_freqs[pp] = 0
            pp_freqs[pp] += 1
    
    total_visit += 1

  dx_probs = dict([(k, v / float(total_visit)) for k, v in dx_freqs.items()
                  ])
  proc_probs = dict([
      (k, v / float(total_visit)) for k, v in proc_freqs.items()
  ])
  dp_probs = dict([(k, v / float(total_visit)) for k, v in dp_freqs.items()
                  ])

  dd_probs = dict([(k, v / float(total_visit)) for k, v in dd_freqs.items()])
    
  pp_probs = dict([(k, v / float(total_visit)) for k, v in pp_freqs.items()])
  
  # Calculate dd and pp cond probs.   
  dd_cond_probs = {}
  pp_cond_probs = {}
    
  for dx1, dx_prob1 in dx_probs.items():
        for dx2, dx_prob2 in dx_probs.items():
            dd = dx1 + ',' + dx2
            if dd in dd_probs:
                dd_cond_probs[dd] = dd_probs[dp] / dx_prob1
            else:
                dd_cond_probs[dd] = 0.0
            
            
   for proc1, proc_prob1 in proc_probs.items():
        for proc2, proc_prob2 in proc_probs.items():
            pp = proc1 + ',' + proc1
            if pp in pp_probs:
                pp_cond_probs[pp] = pp_probs[pp] / proc_prob1
            else:
                pp_cond_probs[dd] = 0.0
            
            
  dp_cond_probs = {}
  pd_cond_probs = {}
  for dx, dx_prob in dx_probs.items():
    for proc, proc_prob in proc_probs.items():
      dp = dx + ',' + proc
      pd = proc + ',' + dx
      if dp in dp_probs:
        dp_cond_probs[dp] = dp_probs[dp] / dx_prob
        pd_cond_probs[pd] = dp_probs[dp] / proc_prob
      else:
        dp_cond_probs[dp] = 0.0
        pd_cond_probs[pd] = 0.0

  pickle.dump(dx_probs, open(output_path + '/dx_probs.empirical.p', 'wb'), -1)
  pickle.dump(proc_probs, open(output_path + '/proc_probs.empirical.p', 'wb'),
              -1)
  pickle.dump(dp_probs, open(output_path + '/dp_probs.empirical.p', 'wb'), -1)
  pickle.dump(dp_cond_probs,
              open(output_path + '/dp_cond_probs.empirical.p', 'wb'), -1)
  pickle.dump(pd_cond_probs,
              open(output_path + '/pd_cond_probs.empirical.p', 'wb'), -1)

  pickle.dump(dd_cond_probs,
              open(output_path + '/dd_cond_probs.empirical.p', 'wb'), -1)
    
  pickle.dump(pp_cond_probs,
              open(output_path + '/pp_cond_probs.empirical.p', 'wb'), -1)

def add_sparse_prior_guide_dp(seqex_list,
                              stats_path,
                              key_set=None,
                              max_num_codes=50):
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
            indices.append(proc1)
            #indices.append((i, max_num_codes + j))
            prob = 0.0 if d not in pp_cond_probs else pp_cond_probs[pp]
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

  return new_seqex_list

def create_patient_objects(seqex_list, path_to_data):
    patients = {}
    for seqex in seqex_list:
        
        pat_id, enc_id = str(seqex.context.feature['patientId'].bytes_list.value[0])[2:].split(':')
        indicies = [x.decode() for x in seqex.feature_lists.feature_list['prior_indices'].feature[0].bytes_list.value] 
        values = [x for x in seqex.feature_lists.feature_list['prior_values'].feature[0].float_list.value] #list(map(list, seqex.feature_lists.feature_list['prior_values'].feature[0].float_list.value))
          
        if pat_id not in patients:
            patients[pat_id] = 0
            pat_temp = Patient(patient_id=pat_id)
        else:
            pat_temp = patients[pat_id]
            
        pat_temp.add_visit(enc_id, indicies, values)
        patients[pat_id] = pat_temp
        
    
    df = pd.DataFrame(columns=['subject_id', 'prior_values', 'prior_indicies'])
    patinfo = []
    for key, item in patients.items():
        info = item.get_information()
        patinfo.append(info)
    
    df = df.append(patinfo, ignore_index=True)
    df.to_parquet(path_to_data + '/prior_table')    
        
    
"""Set <input_path> to where the raw eICU CSV files are located.
Set <output_path> to where you want the output files to be.
"""
def main(argv):
  input_path = argv[1]
  output_path = argv[2]
  num_fold = 1

  patient_file = input_path + '/readmission_data_synthea'
 
  encounter_dict = {}
  print('Processing data')
  encounter_dict = process_patient(patient_file, encounter_dict, hour_threshold=24)
  key_list, seqex_list, dx_map, proc_map = build_seqex(encounter_dict, skip_duplicate=False, min_num_codes=1, max_num_codes=50)

  pickle.dump(dx_map, open(output_path + '/dx_map.p', 'wb'), -1)
  pickle.dump(proc_map, open(output_path + '/proc_map.p', 'wb'), -1)

  stats_path = output_path + '/train_stats'
    
  count_conditional_prob_dp(seqex_list, stats_path)
  train_seqex = add_sparse_prior_guide_dp(seqex_list, stats_path, max_num_codes=50)
  create_patient_objects(train_seqex, input_path) 

if __name__ == '__main__':
    main(sys.argv)
