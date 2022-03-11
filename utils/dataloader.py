from torch.utils.data import DataLoader
from dataset import EHRDataset



# Currently not used

class EHRDataLoader(DataLoader):
    
    def __init__(self, dataset_path):
        
        self.path = dataset_path
    
    def train_dataloader(self):
        
        path = self.path + '/train'
        train_data = EHRDataset(path)
        
        return DataLoader(train_data) 
    
    def validation_dataloader(self):
        
        path = self.path + '/val'
        val_data = EHRDataset(path)
        
        return DataLoader(val_data) 
        
    def test_dataloader(self):
        
        path = self.path + '/test'
        test_data = EHRDataset(path)
        
        return DataLoader(test_data) 