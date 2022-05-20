
import sys
sys.path.insert(1, '../')
from utils.packages import *

global_params = {
    'max_seq_len': 64,
    'gradient_accumulation_steps': 1
}

optim_param = {
    'lr': 3e-5,
    'warmup_proportion': 0.1,
    'weight_decay': 0.01
}

train_params = {
    'batch_size': 256,
    'use_cuda': True,
    'max_len_seq': global_params['max_seq_len'],
    'device': 'cuda' #change this to run on cuda #'cuda:0'
}

def write_voc(data, path):
    
    dcodes = np.concatenate(data['diagnos_code'].tolist(), axis=0)
    diag_codes = np.concatenate(dcodes, axis=0)
    
    if not os.path.isfile(path + 'MLM_diagnoscodes.npy'):
        print("Creating vocabulary for diagnose_codes")
        np.save(path + 'MLM_diagnoscodes.npy', diag_codes)
        
    # Write with medication codes
    med_codes = np.concatenate(np.concatenate(data['medication_code'].tolist(), axis=0), axis=0)
    diag_med_codes = np.append(diag_codes, med_codes)
    if not os.path.isfile(path + 'MLM_diagnosmedcodes.npy'):
        print("Creating vocabulary for diagnose and medication codes")
        np.save(path + 'MLM_diagnosmedcodes.npy', diag_med_codes)
    
    # Write with procedure codes
    proc_codes = np.concatenate(np.concatenate(data['procedure_code'].tolist(), axis=0), axis=0)
    proc_codes = [proc for proc in proc_codes if proc != -1]
    diag_med_proc_codes = np.append(diag_med_codes, proc_codes)
    
    if not os.path.isfile(path + 'MLM_diagnosproccodes.npy'):
        print("Creating vocabulary for diagnose, medication and procedure codes")
        np.save(path + 'MLM_diagnosproccodes.npy', diag_med_proc_codes)
    
    ages = np.concatenate(data['age'].tolist(), axis=0)
    if not os.path.isfile(path + 'MLM_age.npy'):
        print("Creating vocabulary for age")
        np.save(path + 'MLM_age.npy', ages)
        
def load_data(data_name):
    
    path='../data/datasets/' + data_name
    # Split the data if train, test, val does not exist
    #print(path)
    if not os.path.isfile(path + 'train.parquet'):
        raise Exception('train.parquet does not exist, try rerun the process of conditional.py')
    if not os.path.isfile(path + 'test.parquet'):
        raise Exception('test.parquet does not exist, try rerun the process of conditional.py')
    if not os.path.isfile(path + 'val.parquet'):
        raise Exception('val.parquet does not exist, try rerun the process of conditional.py')
        
    train = pd.read_parquet(path + 'train.parquet')
    val = pd.read_parquet(path + 'val.parquet')
    test = pd.read_parquet(path + 'test.parquet')
    
    all_data = pd.concat([train, val, test])
    voc_path = '../data/vocabularies/' + data_name
    write_voc(all_data, voc_path)
    
    return train, val, test

def train_test_model(config, PATH, trainloader, testloader, valloader, tensorboarddir, num_gpus, save_model=False):
    
    trainer = pl.Trainer(
            max_epochs=config['epochs'], 
            gpus=num_gpus,
            plugins='fsdp',
            logger=pl.loggers.TensorBoardLogger(save_dir=tensorboarddir),
            #callbacks=[pl.callbacks.TQDMProgressBar()], #progress.ProgressBar()], 
            progress_bar_refresh_rate=1,
            weights_summary=None, # Can be None, top or full
            num_sanity_val_steps=10,
            precision=16,
        )
    
    conf = BertConfig(config)
    model = BertForMaskedLM(conf) 
    params = list(model.named_parameters())
    optim = adam(params, optim_param)
    
    patienttrajectory = TrainerMLM(model, optim, optim_param, config['reg'], use_prior=config['use_prior'], output_all_encoded_layers=True)
    
    print("Trainer is fitting")
    trainer.fit(
        patienttrajectory, 
        train_dataloaders=trainloader,
        val_dataloaders=valloader,
    );
    print("Predicting on test data")
    predictions = trainer.predict(patienttrajectory, dataloaders=testloader)
    
    avg_acc = sum([ stats['precision'] for stats in predictions ]) / len(predictions)
    avg_acc*100
    
    
    print("Avg precision score: {}".format(avg_acc))
    
    if save_model:
        print("Saving model")
        #PATH = '../saved_models/MLM/CondBEHRT_synthea'
        torch.save(model.state_dict(), PATH)
    
    
def main():
    
    dataset_name = 'Synthea/Final_cohorts/'
    train, val, test = load_data(dataset_name)
    
    feature_types = {'diagnosis':True, 'medications':True, 'procedures':True}
    if (feature_types['diagnosis'] and feature_types['medications'] and not feature_types['procedures']):
        print("Use diagnosis and meds")
        code_voc = 'MLM_diagnosmedcodes.npy'
        age_voc = 'MLM_age.npy'
        
    elif (feature_types['diagnosis'] and not feature_types['medications']):
        print("Use only diagnosis")
        code_voc = 'MLM_diagnoscodes.npy'
        age_voc = 'MLM_age.npy'
        
    else:
        print("Use all features")
        code_voc = 'MLM_diagnosproccodes.npy'
        age_voc = 'MLM_age.npy'
    
    
    files = {'code':'../data/vocabularies/' + dataset_name + code_voc,
             'age':'../data/vocabularies/' + dataset_name + age_voc,
            }
    tokenizer = EHRTokenizer(task='MLM', filenames=files)
    model_config = {
        'vocab_size': len(tokenizer.getVoc('code').keys()), # number of disease + symbols for word embedding
        'hidden_size': 288, #tune.choice([100, 150, 288]), #288, # word embedding and seg embedding hidden size
        'seg_vocab_size': 2, # number of vocab for seg embedding
        'age_vocab_size': len(tokenizer.getVoc('age').keys()), # number of vocab for age embedding,
        'gender_vocab_size': 3,
        'max_position_embeddings': train_params['max_len_seq'], # maximum number of tokens
        'hidden_dropout_prob': 0.1, # dropout rate
        'num_hidden_layers': 6, #4, # number of multi-head attention layers required
        'num_attention_heads': 12, # number of attention heads
        'attention_probs_dropout_prob': 0.1, # multi-head attention dropout rate
        'intermediate_size': 512, # the size of the "intermediate" layer in the transformer encoder
        'hidden_act': 'gelu', # The non-linear activation function in the encoder and the pooler "gelu", 'relu', 'swish' are supported
        'initializer_range': 0.02, # parameter weight initializer range
        'use_prior':True,
        'reg':0.1,
        'age':True,
        'gender':True,
        'epochs':20,
    }
    
    stats_path = '../data/train_stats/Synthea/'
    condfiles = {'dd':stats_path + 'dd_cond_probs.empirical.p', 
                 'dp':stats_path + 'dp_cond_probs.empirical.p',
                 'dm':stats_path + 'dm_cond_probs.empirical.p',
                 'pp':stats_path + 'pp_cond_probs.empirical.p', 
                 'pd':stats_path + 'pd_cond_probs.empirical.p',
                 'pm':stats_path + 'pd_cond_probs.empirical.p',
                 'mm':stats_path + 'mm_cond_probs.empirical.p', 
                 'md':stats_path + 'md_cond_probs.empirical.p',
                 'mp':stats_path + 'mp_cond_probs.empirical.p',
                }
    
    num_gpus = 8
    folderpath = '../data/pytorch_datasets/' + dataset_name
    traind = EHRDataset(train, max_len=train_params['max_len_seq'], feature_types=feature_types, conditional_files=condfiles, save_folder=folderpath, tokenizer=tokenizer, run_type='train_diagnosis_meds_procedures')
    vald = EHRDataset(val, max_len=train_params['max_len_seq'], tokenizer=tokenizer, feature_types=feature_types, save_folder=folderpath, conditional_files=condfiles, run_type='val_diagnosis_meds_procedures')
    testd = EHRDataset(test, max_len=train_params['max_len_seq'], tokenizer=tokenizer, feature_types=feature_types, save_folder=folderpath, conditional_files=condfiles, run_type='test_diagnosis_meds_procedures')
    
  # num_train_examples = 1000  
  #  traind = torch.utils.data.Subset(traind, np.arange(num_train_examples))
  #  vald = torch.utils.data.Subset(vald, np.arange(num_train_examples))
  #  testd = torch.utils.data.Subset(testd, np.arange(num_train_examples))
    
    
    tensorboarddir = '../logs/'
    PATH = '../saved_models/MLM/CondBEHRT_synthea'
    
    trainloader = torch.utils.data.DataLoader(traind, batch_size=train_params['batch_size'], shuffle=False, pin_memory=True, num_workers=4*num_gpus)
    valloader = torch.utils.data.DataLoader(vald, batch_size=train_params['batch_size'], shuffle=False, pin_memory=True, num_workers=4*num_gpus)
    testloader = torch.utils.data.DataLoader(testd, batch_size=train_params['batch_size'], shuffle=False, pin_memory=True, num_workers=4*num_gpus)
    
    train_test_model(model_config, PATH, trainloader, testloader, valloader, tensorboarddir, num_gpus, save_model=True)
    
if __name__=='__main__':
    main()