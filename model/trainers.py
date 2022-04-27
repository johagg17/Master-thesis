import pytorch_lightning as pl
import torch.nn as nn
import torchvision
import numpy as np
import sklearn.metrics as skm
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, confusion_matrix, classification_report, auc
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score, precision_recall_curve
import torch

from torchmetrics import AUROC
from torchmetrics import F1Score

import sys


'''
Trainer for Binary task. 
'''

class TrainerBinaryPrediction(pl.LightningModule):
    
    def __init__(self, model, optim, optim_param):
        '''Put comments here '''
        self.model = model
        self.optim = optim
        self.optim_param = optim_param
        
    def forward(self, age_ids, gender_ids, input_ids, posi_ids, segment_ids, attMask, labels):
        '''Put comments here '''
        return self.model(input_ids, age_ids=age_ids, gender_ids=gender_ids, seg_ids=segment_ids, posi_ids=posi_ids, attention_mask=attMask, labels=labels) 

    def make_prediction(self, batch):
        '''Put comments here'''
        
        age_ids, gender_ids, input_ids, posi_ids, segment_ids, attmask, labels, _ = batch
        loss, pred, labels = self.forward(age_ids, gender_ids, input_ids, posi_ids, segment_ids, attmask, labels)
        
        pred = pred.detach().cpu()
        labels =labels.cpu().type(torch.int32)

        outs = [1 if pred_x >= 0.5 else 0 for pred_x in pred]

        fpr, tpr, threshold = roc_curve(labels.numpy(), pred.numpy(), pos_label=1)
        auc_score = skm.auc(fpr, tpr)

        auc_precision = average_precision_score(labels.numpy(), pred.numpy())
        f1score = f1_score(labels.numpy(), outs)
        
        return (loss, f1score, auc_score, auc_precision)
    
    def training_step(self, batch, batch_idx):
        traingloss, trainf1score, trainaucscore, trainaucprecision = self.make_prediction(batch)
        
        self.log("Training loss", traingloss)
        self.log("Training f1-score", trainf1score)
        self.log("Training AUC", trainaucscore)
        self.log("Training AUCPR", aucprecision)
        
        return {'Training loss': traingloss, 'Training f1-score': trainf1score, 'Training AUC': trainaucscore, 'Training AUCPR': trainaucprecision} 
    
    def validation_step(self, batch, batch_idx):
        valloss, valf1score, valaucscore, valaucprecision = self.make_prediction(batch)
        
        self.log("Validation loss", valloss)
        self.log("Validation f1-score", valf1score)
        self.log("Validation AUC", valaucscore)
        self.log("Validation AUCPR", valaucprecision)
        
        return {'Validation loss': valloss, 'Validation f1-score': valf1score, 'Validation AUC': valaucscore, 'Validation AUCPR': valaucprecision} 
    
    def test_step(self, batch, batch_idx):
        testloss, testf1score, testaucscore, testaucprecision = self.make_prediction(batch)
        
        self.log("Test loss", testloss)
        self.log("Test f1-score", testf1score)
        self.log("Test AUC", testaucscore)
        self.log("Test AUCPR", testaucprecision)
        
        return {'Test loss': valloss, 'Test f1-score': valf1score, 'Test AUC': valaucscore, 'Test AUCPR': valaucprecision} 
    
    def predict_step(self, batch, batch_idx):
        loss, f1score, aucscore, aucprecision = self.make_prediction(batch)
        
        return {'loss': loss, 'f1-score': f1score, 'AUC': aucscore, 'AUCPR': aucprecision} 
        
    
    
    
'''
Trainer for Code prediction
'''


class TrainerCodes(pl.LightningModule):
    
    def __init__(self, model, optim, optim_param, binarizer):
        self.model = model
        self.optim = optim
        self.optim_param = optim_param
        self.mlb = binarizer
        
    
    def forward(self, age_ids, gender_ids, input_ids, posi_ids, segment_ids, attMask, labels):
        '''Put comments here'''
        return self.model(input_ids, age_ids=age_ids, gender_ids=gender_ids, seg_ids=segment_ids, posi_ids=posi_ids, attention_mask=attMask, labels=labels)
    
    
    def make_predictions(self, batch):
        '''Put comments here'''
        age_ids, gender_ids, input_ids, posi_ids, segment_ids, attmask, labels, _ = batch
        
        labels = torch.tensor(self.mlb.transform(labels.cpu().numpy()), dtype=torch.float32).cuda()
        loss, pred, labels = self.forward(age_ids, gender_ids, input_ids, posi_ids, segment_ids, attmask, labels)
        
        sig = nn.Sigmoid()
        output=sig(pred).detach().cpu().numpy()
        labels = labels.cpu().numpy()

        aucpr = average_precision_score(labels, output, average='weighted')
        roc = skm.roc_auc_score(labels,output, average='samples')
                
        return (loss, roc, aucpr) #{'loss': loss, 'Training AUC': roc, 'Training AUCPR':aucpr} 
    
    def training_step(self, batch, batch_idx):
        '''Put comments here'''
       
        trainingloss, trainingAUC, trainingAUCPR = self.make_predictions(batch)
        
        self.log("Training loss", trainingloss)
        self.log("Training AUC", trainingAUC)
        self.log("Training AUCPR", trainingAUCPR)
        
        return {'Training loss': trainingloss, 'Training AUC': trainingAUC, 'Training AUCPR':trainingAUCPR} 
    
    def validation_step(self, batch, batch_idx):
        '''Put comments here'''
        valloss, valAUC, valAUCPR = self.make_predictions(batch)
        
        self.log("Validation loss", valloss)
        self.log("Validation AUC", valAUC)
        self.log("Validation AUCPR", valAUCPR)
        
        return {'Validation loss': valloss, 'Validation AUC': valAUC, 'Validation AUCPR':valAUCPR} 
    
    def test_step(self, batch, batch_idx):
        '''Put comments here'''
        
        testloss, testAUC, testAUCPR = self.make_predictions(batch)
        
        self.log("Test loss", testloss)
        self.log("Test AUC", testAUC)
        self.log("Test AUCPR", testAUCPR)
        
        return {'Test loss': testloss, 'Test AUC': testAUC, 'Test AUCPR':testAUCPR} 
    
    def predict_step(self, batch, batch_idx):
        
        '''Put comments here'''
        
        testloss, testAUC, testAUCPR = self.make_predictions(batch)
        return {'Predict loss': testloss, 'Predict AUC': testAUC, 'Predict AUCPR': testAUCPR} 
    
    

'''
Trainer for MLM
'''

class TrainerMLM(pl.LightningModule):
    ''' LightningModule for training the model using MLM. '''
    
    def __init__(self, model, optim, optim_param):
        '''
        Takes three parameters: 
            model - the model that should be trained
            optim - the optimizer
            optim_param - optimizer parameters
        
        '''
        super(TrainerMLM, self).__init__()
        
        self.model = model
        self.opt = optim
        self.optim_param = optim_param
    
    def compute_acc(self, pred, label):
        '''Put comments here '''
        
        logs = nn.LogSoftmax()
        
        label = label.cpu().numpy()
        
        idx = np.where(label!=-1)[0]
        
        truepred = pred.detach().cpu().numpy()
        
        label = label[idx]
        pred = logs(torch.tensor(truepred[idx]))
        
        outs = [np.argmax(pred_x) for pred_x in pred]
        
        precision = skm.precision_score(label, outs, average='micro')
        
        return precision
    
    def make_prediction(self, batch):
        age_ids, gender_ids, input_ids, posi_ids, segment_ids, attmask, labels, prior_guide = batch
        
        loss, pred, labels = self.forward(age_ids, gender_ids, input_ids, posi_ids, segment_ids, attmask, labels, prior_guide)
                
        precision = self.compute_acc(pred, labels)
        # Compute KL-Divergence loss
        
        
        return (loss, precision)
        
        
    def forward(self, age_ids, gender_ids, input_ids, posi_ids, segment_ids, attMask, labels, prior_guide):
        '''Comment for function '''
        return self.model(input_ids, age_ids=age_ids, gender_ids=gender_ids, seg_ids=segment_ids, posi_ids=posi_ids, attention_mask=attMask, labels=labels, prior_guide=prior_guide)
    
    
    def training_step(self, batch, batch_idx):
        '''Put comments here '''
        #print("batch")
        #print(batch)
        #sys.exit(0)
        #print(batch_idx)
        loss, precision = self.make_prediction(batch)
        
        self.log("Training loss", loss)
        self.log("Training Precision", precision)
        
        return {'loss': loss, 'precision': precision}
    
    def validation_step(self, batch, batch_idx):
        '''Put comments here '''
        loss, precision = self.make_prediction(batch)
        
        self.log("Validation loss", loss)
        self.log("Validation Precision", precision)
        
        return {'loss': loss, 'precision': precision}
    
    def test_step(self, batch, batch_idx):
        '''Put comments here '''
        
        loss, precision = self.make_prediction(batch)
        
        return {'loss': loss, 'precision': precision}
    
    def predict_step(self, batch, batch_idx):
        
        loss, precision = self.make_prediction(batch)
        
        return {'loss': loss, 'precision': precision}
    
    def configure_optimizers(self):
        """Put comments here"""
        # Note: dont use list if only one item.. Causes silent crashes
        #optimizer = torch.optim.Adam(self.model.parameters())
        return self.opt
        
        












































def Accuracy(y_true, y_pred):
    temp = 0
    for i in range(y_true.shape[0]):
        temp += sum(np.logical_and(y_true[i], y_pred[i])) / sum(np.logical_or(y_true[i], y_pred[i]))
    return temp / y_true.shape[0]


class PatientTrajectoryPredictor(pl.LightningModule):
    
    def __init__(self, model, optim, optim_param, train_objective='MLM', metrics=None, binarizer=None):
        
        super().__init__()
        
        self.model = model
        self.opt = optim
        self.train_obj = train_objective
        self.metrics = metrics
        self.mlb = binarizer
        
    def compute_acc(self, pred, label):
        
        logs = nn.LogSoftmax()
        
        label = label.cpu().numpy()
        
        idx = np.where(label!=-1)[0]
        
        truepred = pred.detach().cpu().numpy()
        
        label = label[idx]
        pred = logs(torch.tensor(truepred[idx]))
        
        outs = [np.argmax(pred_x) for pred_x in pred]
        
        precision = skm.precision_score(label, outs, average='micro')
        
        return precision
        
        
    def forward(self, age_ids, gender_ids, input_ids, posi_ids, segment_ids, attMask, labels):
        
        return self.model(input_ids, age_ids = age_ids, gender_ids=gender_ids, seg_ids = segment_ids, posi_ids =
                          posi_ids,attention_mask=attMask, labels=labels)
        
        
    
    
    def training_step(self, batch, batch_idx):
        age_ids, gender_ids, input_ids, posi_ids, segment_ids, attmask, labels, _ = batch
        
        if self.train_obj == 'MLM':
            loss, pred, labels = self.forward(age_ids, gender_ids, input_ids, posi_ids, segment_ids, attmask, labels)
            
        elif self.train_obj == 'visit':
            labels = torch.tensor(self.mlb.transform(labels.cpu().numpy()), dtype=torch.float32).cuda()
            loss, pred, labels = self.forward(age_ids, gender_ids, input_ids, posi_ids, segment_ids, attmask, labels)
        
        elif self.train_obj == 'readmission':
            loss, pred, labels = self.forward(age_ids, gender_ids, input_ids, posi_ids, segment_ids, attmask, labels)
                
        self.log("Training loss", loss)
        
        if self.metrics and self.train_obj == 'MLM':
            m = self.compute_acc(pred, labels)
            self.log("Training Accuracy", m)
            return {'loss': loss, 'Accuracy': m}
        
        elif self.metrics and self.train_obj == 'visit':
            sig = nn.Sigmoid()
            output=sig(pred).detach().cpu().numpy()
            labels = labels.cpu().numpy()
            
            acc = Accuracy(labels, output)
            #tempprc= skm.average_precision_score(labels,output, average='samples')
            
            
            self.log("Training Accuracy", acc)
            return {'loss': loss, 'Acc': acc} 
        
        elif self.train_obj == 'readmission':
            #pred=pred.detach().cpu().numpy()
            #pred = [1 if pr >= 0.5 else 0 for pr in pred]
            #labels = labels.cpu().numpy()
            #acc = np.mean(pred == labels)
            
            #self.log("Training accuracy", acc)
            
            return {'loss': loss}
            
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        age_ids, gender_ids, input_ids, posi_ids, segment_ids, attmask, labels, _ = batch
        if self.train_obj == 'MLM':
            loss, pred, label = self.forward(age_ids, gender_ids, input_ids, posi_ids, segment_ids, attmask, labels)
            
        self.log("Validation loss", loss)
        
        if self.metrics and self.train_obj == 'MLM':
            m = self.compute_acc(pred, label)
            self.log("Validation Accuracy", m)
            return {'loss': loss, 'Accuracy': m}
            
        
    def test_step(self, batch, batch_idx):
        
        age_ids, gender_ids, input_ids, posi_ids, segment_ids, attmask, labels, _ = batch
        loss, pred, labels = self.forward(age_ids, gender_ids, input_ids, posi_ids, segment_ids, attmask, labels)
        
        m = self.compute_acc(pred, labels)
        return {'loss': loss, 'Accuracy': m}
    
    
    def predict_step(self, batch, batch_idx):
        age_ids, gender_ids, input_ids, posi_ids, segment_ids, attmask, labels, _ = batch
        if self.train_obj == 'visit':
            labels = torch.tensor(self.mlb.transform(labels.cpu().numpy()), dtype=torch.float32).cuda()
        loss, pred, labels = self.forward(age_ids, gender_ids, input_ids, posi_ids, segment_ids, attmask, labels)
        
        if self.metrics and self.train_obj == 'visit':
            sig = nn.Sigmoid()
            output=sig(pred).detach().cpu().numpy()
            #output = self.mlb.transform(output)
            labels = labels.cpu().numpy()
            
            aucpr = average_precision_score(labels, output, average='weighted')
            auc_ = 0
            roc = skm.roc_auc_score(labels,output, average='samples')
            #for la, out in zip(labels, output):
             #   auc_ += roc_auc_score(la, out, average=None)
            #auc_ = auc_ / len(labels)
            return {'loss': loss, 'AUC': roc, 'AUCPR':aucpr} 
        if self.train_obj == 'MLM':
            m = self.compute_acc(pred, labels)
            return {'logits': loss, 'Precision':m}
        else:
            pred = pred.detach().cpu()
            labels =labels.cpu().type(torch.int32)
            
            outs = [1 if pred_x > 0.5 else 0 for pred_x in pred]
            
            fpr, tpr, threshold = roc_curve(labels.numpy(), pred.numpy(), pos_label=1)
            auc_score = skm.auc(fpr, tpr)
            
            auc_precision = average_precision_score(labels.numpy(), pred.numpy())
            f1score = f1_score(labels.numpy(), outs)
            
            return {'logits': loss, 'F1-score':f1score, 'AUROC': auc_score, 'AUCPR': auc_precision}
        
    def configure_optimizers(self):
        """ """
        # Note: dont use list if only one item.. Causes silent crashes
        #optimizer = torch.optim.Adam(self.model.parameters())
        return self.opt