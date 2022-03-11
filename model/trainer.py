import pytorch_lightning as pl
import torch.nn as nn
import torchvision
import numpy as np
import sklearn.metrics as skm
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, auc, confusion_matrix, classification_report, auc
import torch

'''
def Accuracy(y_true, y_pred):
    temp = 0
    for i in range(y_true.shape[0]):
        temp += sum(np.logical_and(y_true[i], y_pred[i])) / sum(np.logical_or(y_true[i], y_pred[i]))
    return temp / y_true.shape[0]
'''    

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
            pred=pred.detach().cpu().numpy()
            pred = [1 if pr >= 0.5 else 0 for pr in pred]
            labels = labels.cpu().numpy()
            #print("pred", pred)
            #print("labels", labels)
            acc = np.mean(pred == labels)
            
            self.log("Training accuracy", acc)
            
            return {'loss': loss, 'Training accuracy': acc}
            
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
        #self.log("Training Accuracy", m)
        
        return {'loss': loss, 'Accuracy': m}
    
    
    def predict_step(self, batch, batch_idx):
        age_ids, gender_ids, input_ids, posi_ids, segment_ids, attmask, labels, _ = batch
        loss, pred, labels = self.forward(age_ids, gender_ids, input_ids, posi_ids, segment_ids, attmask, labels)
        m = self.compute_acc(pred, labels)
        return {'logits': loss, 'Accuracy':m}
        
    def configure_optimizers(self):
        """ """
        # Note: dont use list if only one item.. Causes silent crashes
        #optimizer = torch.optim.Adam(self.model.parameters())
        return self.opt