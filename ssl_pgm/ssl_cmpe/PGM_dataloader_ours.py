import numpy as np
import torch
from torch.utils.data import Dataset, default_collate, Subset
from loguru import logger
from torchvision import datasets, transforms
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def replace_negative_values(tensor):
    assert torch.any(tensor != -1), "We need at least one positive value in the bounds"
    logger.info(f"Number of negative values in the tensor: {(tensor == -1).sum().item()}")
    masked_tensor = tensor.clone()
    masked_tensor[masked_tensor == -1] = float('inf')
    masked_tensor[masked_tensor == 0] = float('inf')

    lowest_positive_value = masked_tensor.min().item()
    tensor[tensor == -1] = lowest_positive_value
    tensor[tensor == 0] = lowest_positive_value
    return tensor

class PGM_Dataset(Dataset):
    def __init__(self, X, Y, num_min, num_max, num_sum, denom, device):
        self.device = device
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()
        self.num_min = torch.from_numpy(num_min).to(self.device).float()
        self.num_max = torch.from_numpy(num_max).to(self.device).float()
        self.num_sum = torch.from_numpy(num_sum).to(self.device).float()
        self.denom = torch.from_numpy(denom).to(self.device).float()
        self.denom = replace_negative_values(self.denom)
        self.updated_num = torch.ones(len(self.X)).to(self.device)


    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index]
        num_min = self.num_min[index]
        num_max = self.num_max[index]
        num_sum = self.num_sum[index]
        denom = self.denom[index]
        num_updates = self.updated_num[index]
        # Do any necessary preprocessing or augmentation here
        return x, y, num_min, num_max, num_sum, num_updates, denom, index

    def __len__(self):
        return len(self.X)
    
    def update_bounds(self, new_bounds, num_updates, operation, index):
        self.updated_num[index] = num_updates.to(self.device)
        new_bounds = new_bounds.to(self.device)
        if operation == "max":
            self.num_max[index] = new_bounds
        elif operation == "min":
            self.num_min[index] = new_bounds
        elif operation == "average":
            self.num_sum[index] = new_bounds
        else:
            raise ValueError("Invalid operation")

    
def get_pgm_dataset(dataset, device):
    X_train, y_train = dataset["X_train"], dataset["samples_optimal_y_train"]
    X_test, y_test = dataset["X_test"], dataset["samples_optimal_y_test"]
    X_boolean = torch.tensor(dataset["X_boolean"])
    samples_feasible_y_train, samples_feasible_y_test = dataset["samples_feasible_y_train"], dataset["samples_feasible_y_test"]
    lb_denom_train, lb_denom_test = dataset["lb_denom_train"], dataset["lb_denom_test"]
    initial_ub_num_train, initial_ub_num_test = dataset["initial_ub_num_train"], dataset["initial_ub_num_test"]
    num_X, num_Y = X_train.shape[1], y_train.shape[1]
    return PGM_Dataset(X_train, y_train, initial_ub_num_train, initial_ub_num_train, initial_ub_num_train, lb_denom_train, device), PGM_Dataset(X_test, y_test, initial_ub_num_test, initial_ub_num_test, initial_ub_num_test, lb_denom_test, device), X_boolean, num_X, num_Y
    
def print_range(tensor):
    # Find the minimum and maximum values in the tensor
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)

    # Print the range of the tensor
    print(f"Range of tensor: [{min_val}, {max_val}]")