import numpy as np
import torch
from torch.utils.data import Dataset, default_collate, Subset
from loguru import logger
from torchvision import datasets, transforms
from model_feasible_sol import FeasibleNet


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


class MNISTSubset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        image, label = self.dataset[self.indices[idx]]
        return image, label

    def __len__(self):
        return len(self.indices)


class SubsetDataset(Dataset):
    def __init__(self, subset):
        self.subset = subset

    def __getitem__(self, index):
        return self.subset[index]

    def __len__(self):
        return len(self.subset)


def create_class_datasets(dataset, n_theta, args, select_class, kwargs,
                          class_label=1, device=torch.device('cuda'), ):
    idx = dataset.targets != select_class
    dataset.targets[idx] = 0
    idx2 = dataset.targets != 0
    dataset.targets[idx2] = 1
    class_indices = torch.where(dataset.targets == class_label)[0]
    # Create subsets of the dataset using the class indices
    class_dataset = MNISTSubset(dataset, class_indices)
    indices_for_other_dataset = torch.arange(len(dataset)).long()
    other_dataset = MNISTSubset(dataset, indices_for_other_dataset)
    other_dataset = Subset(other_dataset, torch.where(dataset.targets != class_label)[0])
    print(kwargs)
    train_loader_class_label = torch.utils.data.DataLoader(class_dataset, **kwargs)
    train_loader_other_label = torch.utils.data.DataLoader(other_dataset, **kwargs)
    return train_loader_class_label, train_loader_other_label


def get_MNIST_data_loaders_test(args, n_theta, func_g, train_kwargs, test_kwargs, select_class, class_label=1,
                                device=torch.device('cuda')):
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
    train_loader_class_label, train_loader_other_label = create_class_datasets(train_data, n_theta, args,
                                                                               class_label=class_label,
                                                                               select_class=select_class,
                                                                               kwargs=train_kwargs, device=device)
    test_data = datasets.MNIST('../../data', train=False, transform=transform)
    test_loader_class_label, test_loader_other_label = create_class_datasets(test_data, n_theta, args,
                                                                             class_label=class_label,
                                                                             select_class=select_class,
                                                                             kwargs=test_kwargs, device=device)
    return {f'{class_label}': [train_loader_class_label, test_loader_class_label],
            f'{abs(1 - class_label)}': [train_loader_other_label, test_loader_other_label]}
