import sys
sys.path.insert(1, '../')
from utils.packages import *
import os

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
    
    # Write only diagnose codes
    diag_codes = np.concatenate(data['diagnos_code'].tolist(), axis=0)
    #print("Hej1")
    if not os.path.isfile(path + 'readmission_diagnoscodes.npy'):
        print("Creating vocabulary for diagnose_codes")
        np.save(path + 'readmission_diagnoscodes.npy', diag_codes)
        
    # Write with medication codes
    med_codes = np.concatenate(data['medication_code'].tolist(), axis=0)
    diag_codes = np.append(diag_codes, med_codes)
    if not os.path.isfile(path + 'readmission_diagnosmedcodes.npy'):
        print("Creating vocabulary for diagnose and medication codes")
        np.save(path + 'readmission_diagnosmedcodes.npy', diag_codes)
    
    # Write with procedure codes
    proc_codes = np.concatenate(data['procedure_code'].tolist(), axis=0)
    diag_codes = np.append(diag_codes, proc_codes)
    
    if not os.path.isfile(path + 'readmission_diagnosproccodes.npy'):
        print("Creating vocabulary for diagnose, medication and procedure codes")
        np.save(path + 'readmission_diagnosproccodes.npy', diag_codes)
    
    ages = np.concatenate(data['age'].tolist(), axis=0)
    if not os.path.isfile(path + 'readmission_age.npy'):
        print("Creating vocabulary for age")
        np.save(path + 'readmission_age.npy', ages)
    
def load_data(train_visits, data_name):
    
    path='../data/datasets/' + data_name
    
    if not os.path.isfile(path + 'train.parquet'):
        raise Exception('train.parquet does not exist, try rerun the process of conditional.py')
    if not os.path.isfile(path + 'test.parquet'):
        raise Exception('test.parquet does not exist, try rerun the process of conditional.py')
    if not os.path.isfile(path + 'val.parquet'):
        raise Exception('val.parquet does not exist, try rerun the process of conditional.py')
    train = pd.read_parquet(path + 'train.parquet')
    val = pd.read_parquet(path + 'val.parquet')
    test = pd.read_parquet(path + 'test.parquet')
    
    train = train[train['hadm_id'].map(len) >= train_visits]
    val = val[val['hadm_id'].map(len) >= train_visits]
    test = test[test['hadm_id'].map(len) >= train_visits]
    
    train = train.apply(lambda x: x.str[:train_visits])
    val = val.apply(lambda x: x.str[:train_visits])
    test = test.apply(lambda x: x.str[:train_visits])
    
    all_data = pd.concat([train, val, test])
    
    voc_path = '../data/vocabularies/' + data_name
    write_voc(all_data, voc_path)
    return train, val, test


def train_test_model(config, tokenizer, trainloader, testloader, valloader, tensorboarddir, num_gpus, path_to_model, save_model=False):
    
    
    trainer = pl.Trainer(
            max_epochs=config['epochs'], 
            gpus=num_gpus,
            plugins='fsdp',
            logger=pl.loggers.TensorBoardLogger(save_dir=tensorboarddir),
            callbacks=[pl.callbacks.TQDMProgressBar()], #progress.ProgressBar()], 
            progress_bar_refresh_rate=1,
            weights_summary=None, # Can be None, top or full
            num_sanity_val_steps=10,
            precision=16,
        )

def main():
    
    dataset_name = 'Synthea/Small_cohorts/'
    
    train, val, test = load_data(3, dataset_name)
    
    
    files = {'code':'../data/vocabularies/Synthea/Small_cohorts/diagnosiscodes.npy',
             'age':'../data/vocabularies/Synthea/Small_cohorts/age.npy'
            }    
    
    tokenizer = EHRTokenizer(task='readmission', filenames=files)
    
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
        'use_prior':False,
        'reg':0.1,
        'age':True,
        'gender':False,
        'epochs':20,
    }
    
    '''
    feature_types = {'diagnosis':True, 'medications':False, 'procedures':False}
    num_gpus = 8
    folderpath = '../data/pytorch_datasets/Synthea/Small_cohorts'
    traind = EHRDatasetReadmission(train, max_len=train_params['max_len_seq'], feature_types=feature_types, conditional_files=condfiles, save_folder=folderpath, tokenizer=tokenizer, run_type='train_nextvisit')
    vald = EHRDatasetReadmission(val, max_len=train_params['max_len_seq'], tokenizer=tokenizer, feature_types=feature_types, save_folder=folderpath, conditional_files=condfiles, run_type='val_nextvisit')
    testd = EHRDatasetReadmission(test, max_len=train_params['max_len_seq'], tokenizer=tokenizer, feature_types=feature_types, save_folder=folderpath, conditional_files=condfiles, run_type='test_nextvisit')
    
    tensorboarddir = '../logs/'
    PATH = "../saved_models/MLM/BEHRT_Small_cohort_synthea"
    
    train_test_model(model_config, tokenizer, traind, testd, vald, tensorboarddir, num_gpus, PATH, save_model=True)
    '''
    
if __name__=='__main__':
    main()