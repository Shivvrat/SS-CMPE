import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
    
class PGM_Dataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()

    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index]
        return x, y

    def __len__(self):
        return len(self.X)
    
def get_pgm_dataset(dataset):
    X_train, y_train = dataset["X_train"], dataset["samples_optimal_y_train"]
    X_test, y_test = dataset["X_test"], dataset["samples_optimal_y_test"]
    X_boolean = torch.tensor(dataset["X_boolean"])
    num_X, num_Y = X_train.shape[1], y_train.shape[1]
    return PGM_Dataset(X_train, y_train), PGM_Dataset(X_test, y_test), X_boolean, num_X, num_Y
    
    
