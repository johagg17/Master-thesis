import sys
sys.path.insert(1, '../')
from utils.packages import *
from sklearn.preprocessing import MultiLabelBinarizer

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

def write_voc(data, path, visit_label):
    
    # Write only diagnose codes
    
    label_codes = data['diagnos_code'].apply(lambda x: x[-1]).tolist()
    
    label_codes = np.concatenate(label_codes, axis=0)
    print("Labels")
    print(label_codes)
    #diag_codes = np.concatenate(diag_codes, axis=0)
    
    if not os.path.isfile(path + 'Nextvisit_{}_labelcodes.npy'.format(visit_label)):
        print("Creating vocabulary for labels")
        np.save(path + 'Nextvisit_{}_labelcodes.npy'.format(visit_label), label_codes)

def fix_length(data, visit_label):
    df = data.copy()
    df['hadm_id'] = df['hadm_id'].apply(lambda x: x[:visit_label])
    df['medication_code'] = df['medication_code'].apply(lambda x: x[:visit_label])
    df['diagnos_code'] = df['diagnos_code'].apply(lambda x: x[:visit_label])
    df['procedure_code'] = df['procedure_code'].apply(lambda x: x[:visit_label])
    
    return df

def load_data(data_name, visit_label):
    
    path='../data/datasets/' + data_name
    # Split data if it does not exist
    
    if not os.path.isfile(path + 'train.parquet'):
        raise Exception('train.parquet does not exist, try rerun the process of conditional.py')
    if not os.path.isfile(path + 'test.parquet'):
        raise Exception('test.parquet does not exist, try rerun the process of conditional.py')
    if not os.path.isfile(path + 'val.parquet'):
        raise Exception('val.parquet does not exist, try rerun the process of conditional.py')
        
    train = pd.read_parquet(path + 'train.parquet')
    val = pd.read_parquet(path + 'val.parquet')
    test = pd.read_parquet(path + 'test.parquet')
    
    train = train[train['hadm_id'].map(len) >= visit_label]
    val = val[val['hadm_id'].map(len) >= visit_label]
    test = test[test['hadm_id'].map(len) >= visit_label]
    
    train = fix_length(train, visit_label)
    val = fix_length(val, visit_label)
    test = fix_length(test, visit_label)
    
    all_data = pd.concat([train, val, test])
    
    voc_path = '../data/vocabularies/' + data_name
    write_voc(all_data, voc_path, visit_label)
    
    return train, val, test


def train_test_model(config, tokenizer, mlb, trainloader, testloader, valloader, tensorboarddir, num_gpus, path_to_model, save_path, save_model=False):
    
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
    model = BertMultiLabelPrediction(conf, num_labels=len(tokenizer.getVoc('label').keys())) 
    model = load_model(path_to_model, model)
    params = list(model.named_parameters())
    optim = adam(params, optim_param)
    
    patienttrajectory = TrainerCodes(model, optim, optim_param, binarizer=mlb)
    
    print("Trainer is fitting")
    trainer.fit(
        patienttrajectory, 
        train_dataloaders=trainloader,
        val_dataloaders=valloader,
    );
    print("Predicting on test data")
    predictions = trainer.predict(patienttrajectory, dataloaders=testloader)
    
    avg_auc = sum([ stats['AUC'] for stats in predictions ]) / len(predictions)
    avg_aucpr = sum([ stats['AUCPR'] for stats in predictions ]) / len(predictions)
     
    print("Avg AUC: {}".format(avg_auc*100))
    print("Avg AUCPR: {}".format(avg_aucpr*100))
    
    if save_model:
        print("Saving model")
        torch.save(model.state_dict(), save_path)

def main():
    
    
    #dataset_name = 'Synthea/Small_cohorts/'
    dataset_name = 'MIMIC/'
    labelvisit = 8
    visits_to_train_on = 7
    train, val, test = load_data(dataset_name, labelvisit)
    
    feature_types = {'diagnosis':True, 'medications':False, 'procedures':False}
    if (feature_types['diagnosis'] and feature_types['medications']):
        print("Do only use diagnosis")
        code_voc = 'MLM_diagnosmedcodes.npy'
        
    elif (feature_types['diagnosis'] and not feature_types['medications']):
        code_voc = 'MLM_diagnoscodes.npy'
    else:
        code_voc = 'MLM_diagnosmedproccodes.npy'
        
        
    age_voc = 'MLM_age.npy'
    label_voc = 'Nextvisit_{}_labelcodes.npy'.format(labelvisit)
    
    files = {'code':'../data/vocabularies/' + dataset_name + code_voc,
             'age':'../data/vocabularies/' + dataset_name + age_voc,
             'labels':'../data/vocabularies/' + dataset_name + label_voc,
            }
    
    tokenizer = EHRTokenizer(task='nextvisit', filenames=files)
    
    mlb = MultiLabelBinarizer(classes=list(tokenizer.getVoc('label').values()))
    mlb.fit([[each] for each in list(tokenizer.getVoc('label').values())])
    
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
        'epochs':15,
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
    
    feature_types = {'diagnosis':True, 'medications':False, 'procedures':False}
    num_gpus = 8
    folderpath = '../data/pytorch_datasets/' + dataset_name
    
    traind = EHRDatasetCodePrediction(train, max_len=train_params['max_len_seq'], feature_types=feature_types, conditional_files=condfiles, train_visits=visits_to_train_on,labelvisit=labelvisit, save_folder=folderpath, tokenizer=tokenizer, run_type='train_nextvisit')
    vald = EHRDatasetCodePrediction(val, max_len=train_params['max_len_seq'], tokenizer=tokenizer, train_visits=visits_to_train_on,labelvisit=labelvisit, feature_types=feature_types, save_folder=folderpath, conditional_files=condfiles, run_type='val_nextvisit')
    testd = EHRDatasetCodePrediction(test, max_len=train_params['max_len_seq'], tokenizer=tokenizer, feature_types=feature_types, train_visits=visits_to_train_on, labelvisit=labelvisit, save_folder=folderpath, conditional_files=condfiles, run_type='test_nextvisit')
    
   # num_train_examples = 1000
    
  #  traind = torch.utils.data.Subset(traind, np.arange(num_train_examples))
  #  vald = torch.utils.data.Subset(vald, np.arange(num_train_examples))
  #  testd = torch.utils.data.Subset(testd, np.arange(num_train_examples))
    
    
    tensorboarddir = '../logs/'
    
    trainloader = torch.utils.data.DataLoader(traind, batch_size=train_params['batch_size'], shuffle=False, pin_memory=True, num_workers=4*num_gpus)
    valloader = torch.utils.data.DataLoader(vald, batch_size=train_params['batch_size'], shuffle=False, pin_memory=True, num_workers=4*num_gpus)
    testloader = torch.utils.data.DataLoader(testd, batch_size=train_params['batch_size'], shuffle=False, pin_memory=True, num_workers=4*num_gpus)
    
    #PATH = '../saved_models/MLM/BEHRT_Synthea'
    PATH = '../saved_models/MLM/BEHRT_mimic'
    save_path = '../saved_models/NextxVisit/BEHRT_mimic_NextVisit{}_trainvisits{}_'.format(labelvisit, visits_to_train_on)
    train_test_model(model_config, tokenizer, mlb, trainloader, testloader, valloader, tensorboarddir, num_gpus, PATH, save_path, save_model=True)
    
if __name__=='__main__':
    main()
    
    
    