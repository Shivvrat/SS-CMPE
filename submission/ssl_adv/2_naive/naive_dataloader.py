import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
    
class ILP(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()

    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index]
        return x, y

    def __len__(self):
        return len(self.X)
    
def get_ilp_dataset(ilp_data_location):
    data = np.load(ilp_data_location)
    X, y = data["initial"], data["updated"]
    return ILP(X, y)
    
    
