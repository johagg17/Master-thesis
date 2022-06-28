import sys
sys.path.insert(1, '../')
from utils.packages import *
import os

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

def get_conf(tokenizer):
    
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
    

    return model_config

def train_test_model(config, tokenizer, trainloader, testloader, modelname, tensorboarddir, num_gpus, path_to_model, save_path, text_file_path, mlb, save_model=False):
    
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
    
    trainer.fit(
        patienttrajectory, 
        train_dataloaders=trainloader,
    );
    
    print("Predicting on test data")
    predictions = trainer.predict(patienttrajectory, dataloaders=testloader)
    
    avg_auc = sum([ stats['AUC'] for stats in predictions ]) / len(predictions)
    print("Avg_AUC {}".format(avg_auc*100))
    
    avg_aucpr = sum([ stats['AUCPR'] for stats in predictions ]) / len(predictions)
    print("Avg_AUCPR {}".format(avg_aucpr*100))
    
    with open(text_file_path + '/auc_{}.txt'.format(modelname), 'a') as f:
        txt = str(avg_auc) + ','
        f.write(txt)
    
    with open(text_file_path + '/aucpr_{}.txt'.format(modelname), 'a') as f:
        txt = str(avg_aucpr) + ','
        f.write(txt)
    
    if save_model:
        print("Saving model")
        torch.save(model.state_dict(), save_path)
    
    
def main(dataset_name, modelname):
    
    nfolds = 3
    
    foldpath = '../data/cross_val/{}'.format(dataset_name)
    
    feature_types = {'diagnosis':True, 'medications':False, 'procedures':False}
    if (feature_types['diagnosis'] and feature_types['medications'] and not feature_types['procedures']): # Use diagnosis and meds
        print("Use diagnosis and meds")
        code_voc = 'MLM_diagnosmedcodes.npy' # Voc för diagnos och meds
        age_voc = 'MLM_age.npy'
    
    elif (feature_types['diagnosis'] and feature_types['procedures'] and not feature_types['medications']): # Use diagnosis and procedures
        print("Use only diagnosis and procedures")
        code_voc = 'MLM_diagnosnotmedproccodes.npy'
        age_voc = 'MLM_age.npy'
        
        
    elif (feature_types['diagnosis'] and not (feature_types['medications'] and feature_types['procedures'])): # Use only diagnosis
        print("Use only diagnosis")
        code_voc = 'MLM_diagnoscodes.npy' # Voc för endast diagnoser
        age_voc = 'MLM_age.npy'
    else:
        print("Use all features")
        code_voc = 'MLM_diagnosproccodes.npy' # Voc för diagnoser, procedures, medications
        age_voc = 'MLM_age.npy'
    
    if dataset_name == 'Synthea':
        files = {'code':'../data/vocabularies/' + dataset_name  + '/Final_cohorts/'+ code_voc,
                 'age':'../data/vocabularies/' + dataset_name + '/Final_cohorts/' +  age_voc,
                 'labels':''
                }
    else:
        files = {'code':'../data/vocabularies/' + dataset_name  + '/'+ code_voc,
                 'age':'../data/vocabularies/' + dataset_name + '/' +  age_voc,
                 'labels':''
                }
        
    
    if dataset_name == 'MIMIC':
        stats_path = '../data/train_stats/MIMIC2/'
    else:
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
    nextvisit = 5
    
    #for fold_idx in range(nfolds):
        
    current_fold = 3 #fold_idx + 1

    folderpath = '../data/cross_val/' + dataset_name + '/nextvisit{}/fold{}'.format(nextvisit, current_fold)

    textfilepath = '../data/cross_val/' + dataset_name + '/nextvisit{}/fold{}'.format(nextvisit, current_fold)

    train = pd.read_parquet(folderpath+ '/train.parquet')
    test = pd.read_parquet(folderpath + '/test.parquet')

    labelvoc = folderpath + '/Nextvisit_{}_labelcodes_fold{}.npy'.format(nextvisit, current_fold)
    files['labels'] = labelvoc
    
    tokenizer = EHRTokenizer(task='nextvisit', filenames=files)

    mlb = MultiLabelBinarizer(classes=list(tokenizer.getVoc('label').values()))
    mlb.fit([[each] for each in list(tokenizer.getVoc('label').values())])


    config = get_conf(tokenizer)
    
    visits_to_train_on = 4
    
    traind = EHRDatasetCodePrediction(train, max_len=train_params['max_len_seq'], feature_types=feature_types, conditional_files=condfiles,
                                      train_visits=visits_to_train_on,labelvisit=nextvisit, save_folder=folderpath, tokenizer=tokenizer,
                                      run_type='train_nextvisit_d')

    testd = EHRDatasetCodePrediction(test, max_len=train_params['max_len_seq'], tokenizer=tokenizer, feature_types=feature_types,
                                     train_visits=visits_to_train_on, labelvisit=nextvisit, save_folder=folderpath, conditional_files=condfiles,
                                     run_type='test_nextvisit_d')
    
    trainloader = torch.utils.data.DataLoader(traind, batch_size=train_params['batch_size'], shuffle=False, pin_memory=True,num_workers=4*num_gpus)
    testloader = torch.utils.data.DataLoader(testd, batch_size=train_params['batch_size'], shuffle=False, pin_memory=True, num_workers=4*num_gpus)

    PATH = '../saved_models/MLM/{}_{}'.format(modelname, dataset_name.lower())
    tensorboarddir = '../logs/'
    
    save_path = '../saved_models/NextxVisit/{}_{}_NextVisit{}_trainvisits{}_fold{}'.format(modelname, dataset_name.lower(), nextvisit, nextvisit, current_fold)
    train_test_model(config, tokenizer, trainloader, testloader, modelname, tensorboarddir, num_gpus, PATH, save_path, textfilepath, mlb, save_model=True)
        
        
if __name__=='__main__':
    dataname = 'MIMIC' #'Synthea' #'MIMIC'
    modelname = 'CondBEHRT_-p-m' #'CondBEHRT_-p-m' #'CondBEHRT'
    
    main(dataname, modelname)