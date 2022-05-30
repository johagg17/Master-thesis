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

def fix_length(data, visit_label):
    df = data.copy()
    df['hadm_id'] = df['hadm_id'].apply(lambda x: x[:visit_label])
    df['medication_code'] = df['medication_code'].apply(lambda x: x[:visit_label])
    df['diagnos_code'] = df['diagnos_code'].apply(lambda x: x[:visit_label])
    df['procedure_code'] = df['procedure_code'].apply(lambda x: x[:visit_label])
    
    return df    

def get_balanced_data(train, val, test, label_visit):
    
    train['visit_label'] = train['label'].apply(lambda x: x[label_visit - 1])
    train_0= train[train['visit_label'] == 0]
    train_1 = train[train['visit_label'] == 1]
    
    if len(train_0) > len(train_1):
        train_0 = train_0.sample(len(train_1))
    else:
        train_1 = train_1.sample(len(train_0))
        
    train = pd.concat([train_0, train_1])
    
    val['visit_label'] = val['label'].apply(lambda x: x[label_visit - 1])
    val_0= val[val['visit_label'] == 0]
    val_1 = val[val['visit_label'] == 1]
    if len(val_0) > len(val_1):
        val_0 = val_0.sample(len(val_1))
    else:
        val_1 = val_1.sample(len(val_0))
        
    val = pd.concat([val_0, val_1])
    
    test['visit_label'] = test['label'].apply(lambda x: x[label_visit - 1])
    test_0= test[test['visit_label'] == 0]
    test_1 = test[test['visit_label'] == 1]
    
    if len(test_0) > len(test_1):
        test_0 = test_0.sample(len(test_1))
    else:
        test_1 = test_1.sample(len(test_0))
        
    test = pd.concat([test_0, test_1])
    
    return train, val, test
    
    
def load_data(label_visit, data_name, balance_data=False):
    
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
    
    train = train[train['hadm_id'].map(len) >= label_visit]
    val = val[val['hadm_id'].map(len) >= label_visit]
    test = test[test['hadm_id'].map(len) >= label_visit]
    
    train = fix_length(train, label_visit)
    val = fix_length(val, label_visit)
    test = fix_length(test, label_visit)
    
    if balance_data:
        train, val, test = get_balanced_data(train, val, test, label_visit)
    
    return train, val, test


def train_test_model(config, tokenizer, trainloader, testloader, valloader, tensorboarddir, num_gpus, path_to_model, save_path, save_model=False):
    
    
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
    
    conf = BertConfig(config)
    model = BertSinglePrediction(conf, num_labels=1) 
   # PATH = "../saved_models/MLM/model_with_prior_82test"
    model = load_model(path_to_model, model)
    params = list(model.named_parameters())
    optim = adam(params, optim_param)
    
    patienttrajectory = TrainerBinaryPrediction(model, optim, optim_param)

    print("Trainer is fitting")
    trainer.fit(
        patienttrajectory, 
        train_dataloaders=trainloader,
        val_dataloaders=valloader,
    );
    print("Predicting on test data")
    predictions = trainer.predict(patienttrajectory, dataloaders=testloader)
    
    avg_f1 = sum([ stats['f1-score'] for stats in predictions ]) / len(predictions)
    print("Avg_F1 {}".format( avg_f1*100))
    
    avg_auc = sum([ stats['AUC'] for stats in predictions ]) / len(predictions)
    print("Avg_AUC {}".format(avg_auc*100))
    
    avg_aucpr = sum([ stats['AUCPR'] for stats in predictions ]) / len(predictions)
    print("Avg_AUCPR {}".format(avg_aucpr*100))
    
    if save_model:
        print("Saving model")
        torch.save(model.state_dict(), save_path)

def main():
    
    dataset_name = 'MIMIC/'#'Synthea/Final_cohorts/'
    readmissionvisit = 10
    visits_to_train_on = 10
    balanced_data = False # 
    train, val, test = load_data(readmissionvisit, dataset_name, balanced_data)
    
    #print(len(train))
    #print(len(val))
    #print(len(test))
    
    feature_types = {'diagnosis':True, 'medications':False, 'procedures':False}
    if (feature_types['diagnosis'] and feature_types['medications'] and not feature_types['procedures']):
        print("Do only use diagnosis")
        code_voc = 'MLM_diagnosmedcodes.npy'
        
    elif (feature_types['diagnosis'] and not feature_types['medications']):
        code_voc = 'MLM_diagnoscodes.npy'
    else:
        code_voc = 'MLM_diagnosproccodes.npy'
        
        
    age_voc = 'MLM_age.npy'
    
    files = {'code':'../data/vocabularies/' + dataset_name + code_voc,
             'age':'../data/vocabularies/' + dataset_name + age_voc
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
        'use_prior':True,
        'reg':0.1,
        'age':True,
        'gender':True,
        'epochs':15,
    }
    
    stats_path = '../data/train_stats/MIMIC2/'
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
    traind = EHRDatasetReadmission(train, label_visit=readmissionvisit, nvisits=visits_to_train_on, max_len=train_params['max_len_seq'], feature_types=feature_types, conditional_files=condfiles, save_folder=folderpath, tokenizer=tokenizer, run_type='train_readmission_d_balanced{}'.format(balanced_data))
    vald = EHRDatasetReadmission(val, label_visit=readmissionvisit, nvisits=visits_to_train_on, max_len=train_params['max_len_seq'], tokenizer=tokenizer, feature_types=feature_types, save_folder=folderpath, conditional_files=condfiles, run_type='val_readmission_d_balanced{}'.format(balanced_data))
    testd = EHRDatasetReadmission(test, label_visit=readmissionvisit, nvisits=visits_to_train_on, max_len=train_params['max_len_seq'], tokenizer=tokenizer, feature_types=feature_types, save_folder=folderpath, conditional_files=condfiles, run_type='test_readmission_d_balanced{}'.format(balanced_data))
    
    trainloader = torch.utils.data.DataLoader(traind, batch_size=train_params['batch_size'], shuffle=False, pin_memory=True,num_workers=4*num_gpus)
    valloader = torch.utils.data.DataLoader(vald, batch_size=train_params['batch_size'], shuffle=False, pin_memory=True, num_workers=4*num_gpus)
    testloader = torch.utils.data.DataLoader(testd, batch_size=train_params['batch_size'], shuffle=False, pin_memory=True, num_workers=4*num_gpus)
    
    tensorboarddir = '../logs/'
    PATH = '../saved_models/MLM/CondBEHRT_d_mimic'
    save_path = '../saved_models/Readmission/CondBEHRT_d_mimic_visits{}_labelvisit{}_balanced{}'.format(readmissionvisit, visits_to_train_on, balanced_data)
    train_test_model(model_config, tokenizer, trainloader, testloader, valloader, tensorboarddir, num_gpus, PATH, save_path, save_model=True)
    
    
if __name__=='__main__':
    main()