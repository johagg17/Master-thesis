import sys
sys.path.insert(1, '../')

from utils.dataset import EHRDataset
from model.tokenizer import EHRTokenizer
import pytorch_pretrained_bert as Bert
from torch.utils.data import DataLoader
from model.model import *
from utils.config import *
from utils.optimizer import adam
from model.trainers import TrainerMLM
import pytorch_lightning as pl
from sklearn.model_selection import KFold
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from utils.vocabulary import *

warnings.filterwarnings('ignore')






def main():
    path = '../data/datasets/readmission_data_synthea'
    
    global_params = {
        'max_seq_len': 256,
        'gradient_accumulation_steps': 1
    }

    optim_param = {
        'lr': 3e-5,
        'warmup_proportion': 0.1,
        'weight_decay': 0.01
    }

    train_params = {
        'batch_size': 32,
        'use_cuda': True,
        'max_len_seq': global_params['max_seq_len'],
        'device': 'cuda' #change this to run on cuda #'cuda:0'
    }
    
    data = pd.read_parquet(path)
    data['subject_id'] = data['subject_id'].apply(lambda x: x.replace('-', ''))
    
    files = {'code':'../data/vocabularies/Synthea/snomed.npy','age':'../data/vocabularies/Synthea/age.npy'}
    tokenizer = EHRTokenizer(task='MLM', filenames=files)
    
    model_config = {
        'vocab_size': len(tokenizer.getVoc('code').keys()), # number of disease + symbols for word embedding
        'hidden_size': 288, # word embedding and seg embedding hidden size
        'seg_vocab_size': 2, # number of vocab for seg embedding
        'age_vocab_size': len(tokenizer.getVoc('age').keys()), # number of vocab for age embedding,
        'gender_vocab_size': 3,
        'max_position_embeddings': train_params['max_len_seq'], # maximum number of tokens
        'hidden_dropout_prob': 0.1, # dropout rate
        'num_hidden_layers': 2, # number of multi-head attention layers required
        'num_attention_heads': 4, # number of attention heads
        'attention_probs_dropout_prob': 0.1, # multi-head attention dropout rate
        'intermediate_size': 288, # the size of the "intermediate" layer in the transformer encoder
        'hidden_act': 'gelu', # The non-linear activation function in the encoder and the pooler "gelu", 'relu', 'swish' are supported
        'initializer_range': 0.02, # parameter weight initializer range
    }
    
    conf = BertConfig(model_config)
    tensorboarddir = '../logs/'
    
    trainer = pl.Trainer(
            max_epochs=5, 
            gpus=-1,
            logger=pl.loggers.TensorBoardLogger(save_dir=tensorboarddir),
            callbacks=[pl.callbacks.TQDMProgressBar()], 
            progress_bar_refresh_rate=1,
            #strategy='DeepSpeed',
            weights_summary=None, # Can be None, top or full
            num_sanity_val_steps=10,
        )
    
    trainset, testset = train_test_split(data, test_size=0.2)
    traind = EHRDataset(trainset, max_len=train_params['max_len_seq'], tokenizer=tokenizer)
    testd = EHRDataset(testset, max_len=train_params['max_len_seq'], tokenizer=tokenizer)
    trainloader = torch.utils.data.DataLoader(traind, batch_size=32, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testd, batch_size=32, shuffle=True, num_workers=4)
    
    model = BertForMaskedLM(conf) #BertForMaskedLM(conf)
    params = list(model.named_parameters())
    optim = adam(params, optim_param)
    
    patienttrajectory = TrainerMLM(model, optim, optim_param)

    trainer.fit(
        patienttrajectory, 
        train_dataloaders=trainloader,
    );

    predictions = trainer.predict(patienttrajectory, dataloaders=testloader)
    
    avg_acc = sum([ stats['precision'] for stats in predictions ]) / len(predictions)
    avg_acc*100
    
    
    

main()