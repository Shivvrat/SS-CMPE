import numpy as np
import torch
from torch.utils.data import Dataset, default_collate, Subset
from loguru import logger
from torchvision import datasets, transforms
from model_feasible_sol import FeasibleNet
from lp.lp_intial_P_ub import get_initial_bounds_for_num, get_denom_for_lambda
# def select_correctly_predicted_examples(model, dataset, args, device=torch.device('cuda')):
#     correct_pred_indices = []
#     for i, data in enumerate(dataset):
#         if args.debug and i == 5:
#             break
#         inputs, labels = data
#         inputs = inputs.reshape(-1, 28 * 28).to(device)
#         outputs = model(inputs.to(device))
#         output = torch.sigmoid(outputs)
#         # print(output)
#         pred = (output > 0.5).float()
#         # print(pred, labels, args.class_index)
#         if pred == int(not args.not_class):
#             correct_pred_indices.append(i)
#     correct_pred_indices = torch.LongTensor(correct_pred_indices)
#     print(correct_pred_indices.shape)
#     return correct_pred_indices

def print_range(tensor):
    # Find the minimum and maximum values in the tensor
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)

    # Print the range of the tensor
    print(f"Range of tensor: [{min_val}, {max_val}]")

class MNISTSubset(Dataset):
    def __init__(self, dataset, indices, device):
        self.device = device
        self.images = dataset.data[indices].detach().clone().numpy()
        self.targets = dataset.targets[indices]
        self.num_min = torch.ones(len(dataset)).to(self.device)
        self.num_max = torch.ones(len(dataset)).to(self.device)
        self.num_sum = torch.ones(len(dataset)).to(self.device)
        self.denom = torch.ones(len(dataset)).to(self.device)
        # Store this to see if initial update has been done or not
        # First update is always done (regardless of the value of the bound)
        # Count number of updates for the average - average = sum/num_updates
        self.updated_num = torch.ones(len(dataset)).to(self.device)
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transforms:
            image = self.transforms(image).float()
        label = self.targets[idx].float()
        num_min = self.num_min[idx].float()
        num_max = self.num_max[idx].float()
        num_sum = self.num_sum[idx].float()
        denom = self.denom[idx].float()
        num_updates = self.updated_num[idx]
        # Use idx here to update the bounds
        return image, label, num_min, num_max, num_sum, num_updates, denom, idx

    def __len__(self):
        return len(self.images)
    
    def add_denom_and_num(self, num_min, denom):
        self.num_min = num_min.clone().detach().to(self.device)
        self.num_max = num_min.clone().detach().to(self.device)
        self.num_sum = num_min.clone().detach().to(self.device)
        self.denom = denom.clone().detach().to(self.device)
        
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

def create_class_datasets(dataset, n_theta, nn_as_dict_of_np_array, f_xy,  args, select_class, kwargs,
                          class_label=1):
    idx = dataset.targets != select_class
    dataset.targets[idx] = 0
    idx2 = dataset.targets != 0
    dataset.targets[idx2] = 1
    class_indices = torch.where(dataset.targets == class_label)[0]
    # Create subsets of the dataset using the class indices
    class_dataset = MNISTSubset(dataset, class_indices, args.device)
    
    # Get bounds for warm start
    initial_bounds_num = get_initial_bounds_for_num(class_dataset, nn_as_dict_of_np_array, f_xy, args)
    logger.info(f"Range for initial bounds for num: {torch.min(initial_bounds_num)}, {torch.max(initial_bounds_num)}")
    # Get denominator values for lambda
    denom_values_for_lambda = get_denom_for_lambda(class_dataset, nn_as_dict_of_np_array, f_xy, n_theta, args)
    logger.info(f"Range for bounds for denom: {torch.min(denom_values_for_lambda)}, {torch.max(denom_values_for_lambda)}")
    
    print(print_range(initial_bounds_num))
    print(print_range(denom_values_for_lambda))
    logger.info("Updated the bounds for the class dataset")
    print(kwargs)
    return class_dataset


def get_MNIST_data_loaders(args, n_theta, nn_as_dict_of_np_array, f_xy, train_kwargs, test_kwargs, select_class, class_label,
                                device=torch.device('cuda')):
    func_g = n_theta
    if args.binarize:
        logger.info("Binarizing the dataset")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            lambda x: x > 0,
            lambda x: x.float(),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,)),
        ])
    train_data = datasets.MNIST('../../data', train=True, download=True,
                                transform=transform)
    input_dim = train_data.data.shape[1]*train_data.data.shape[2]
    # create a Subset object using the indices tensor
    train_dataset_class_label = create_class_datasets(train_data, n_theta, nn_as_dict_of_np_array, f_xy, args,
                                                                               class_label=class_label,
                                                                               select_class=select_class,
                                                                               kwargs=train_kwargs)
    test_data = datasets.MNIST('../../data', train=False, transform=transform)
    test_dataset_class_label = create_class_datasets(test_data, n_theta, nn_as_dict_of_np_array, f_xy, args,
                                                                             class_label=class_label,
                                                                             select_class=select_class,
                                                                             kwargs=test_kwargs)
    return {f'{class_label}': [train_dataset_class_label, test_dataset_class_label]}
