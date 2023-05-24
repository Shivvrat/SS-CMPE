from __future__ import print_function
import argparse
import copy
import os
import math
from pprint import pprint
import time
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Module
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
import wandb
from loguru import logger
from torch.utils.data import Dataset, default_collate, Subset
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import save_image
from ilp_runner import run_ilp

from utils import create_directory, get_date_as_string


# Get the parent directory path
parent_dir = os.path.join(os.path.dirname(os.getcwd()), "train_classifier")

# Append the parent directory path to system path
sys.path.append(parent_dir)

from MNIST_dataloader import get_MNIST_data_loaders_test

class Net(nn.Module):
    def __init__(self, input_size, layers_data: list, ):
        super(Net, self).__init__()
        self.layers = nn.ModuleList()
        self.input_size = input_size
        for size, activation in layers_data:
            self.layers.append(nn.Linear(input_size, size))
            input_size = size  # For the next layer
            if activation is not None:
                assert isinstance(activation, Module), \
                    "Each tuples should contain a size (int) and a torch.nn.modules.Module."
                self.layers.append(activation)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class LR(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LR, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        return out

def generate_output_images(updated_images, initial_images, device, output_dir):
    if not os.path.exists(output_dir):
        # If the folder does not exist, create it
        os.makedirs(output_dir)
        print("Folder created successfully!")
    else:
        print("Folder already exists!")
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            # check if the file is a regular file (not a directory)
            if os.path.isfile(file_path):
                os.remove(file_path)  # delete the file
    save_idx = 0
    logger.info("Generating output images")
    logger.info(f"{output_dir}")
    for updated_image, initial_image in zip(updated_images, initial_images):
        updated_image = torch.from_numpy(updated_image.reshape(28, 28))
        initial_image = torch.from_numpy(initial_image.reshape(28, 28))
        concatenated_image = torch.cat((initial_image, updated_image), dim=1)
        save_image(concatenated_image, f'{output_dir}/img_{save_idx}.png')
        save_idx += 1


def load_nn(nn_path, use_cuda=None):
    """
    Load the NN model given its path.
    :param nn_path: Location of the saved model
    :return: Loaded model
    """
    if not use_cuda:
        use_cuda = torch.cuda.is_available()
    if use_cuda:
        nn = torch.load(nn_path)
    else:
        nn = torch.load(nn_path, map_location=torch.device('cpu'))
    nn_as_dict_of_np_array = {}
    for param_tensor in nn.state_dict():
        logger.info(param_tensor, "\t", nn.state_dict()[param_tensor].size())
        nn_as_dict_of_np_array[f"{param_tensor}"] = nn.state_dict()[param_tensor].cpu().detach().numpy()
    return nn_as_dict_of_np_array, nn


def ilp_solver(args, model=None):
    debug = False
    args.debug = debug
    class_label = int(not args.not_class)
    select_class = args.class_index
    if not debug and model is None:
        wandb.init(project=f"ilp_{select_class}_{class_label}")
        wandb.config.update(args)
    if not args.not_class:
        name = f"{args.func_f}/{args.class_index}/{args.class_index}"
    else:
        name = f"{args.func_f}/{args.class_index}/not_{args.class_index}"
    if args.binarize:
        output_dir = f"models/binarized{args.func_f}/{name}"
    else:
        output_dir = f"models/continous/{args.func_f}/{name}"
    # Add a logger to the project

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # Load the LR Model
    if model is None:
        if not args.use_nn:
            n_theta_as_dict_of_np_array, n_theta = load_nn(args.model,
                                                        use_cuda)

        # Load the large model
        else:
            n_theta_as_dict_of_np_array, n_theta = load_nn(args.model, use_cuda)
    else:
        n_theta = model
    logger.info(f"You have selected {select_class}")
    logger.info(f"You have selected {class_label}")
    if args.binarize:
        logger.info(f"You have selected binarized dataset")
    else:
        logger.info(f"You have selected continous dataset")
    
    if args.dataset.lower() == 'mnist':
        loaders = get_MNIST_data_loaders_test(args, n_theta, n_theta, train_kwargs, test_kwargs,
                                              select_class=select_class, class_label=class_label, device=device)
    elif args.dataset.lower() == 'emnist':
        train_loader, test_loader = get_EMNIST_data_loaders(args, train_kwargs, test_kwargs, select_class=9)
    if not args.not_class:
        loader_for_1 = loaders['1']
        train_loader, test_loader = loader_for_1
    else:
        loader_for_0 = loaders['0']
        train_loader, test_loader = loader_for_0
    input_batch, _ = next(iter(train_loader))
    # extract the number of features
    logger.info(f"{input_batch.shape[2]}, {input_batch.shape[3]}")
    args.input_size = input_batch.shape[2] * input_batch.shape[3]
    date = get_date_as_string()
    updated, initial, missed_examples = run_ilp(debug, 64, n_theta_as_dict_of_np_array, args, train_loader, date, name, dataset="train")
    updated, initial, missed_examples = run_ilp(debug, 64, n_theta_as_dict_of_np_array, args, test_loader, date, name, dataset="test")
    # Save Actual and Adversarial Images
    if args.binarize:
        generate_output_images(updated, initial, device, f"adv_images_binarized/{args.func_f}/{name}")
    else:
        generate_output_images(updated, initial, device, f"adv_images_continous/{args.func_f}/{name}")
               


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='SS-CMPE Experiments')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=2048, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--ae-epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.9, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--binarize', action='store_true', default=False,
                        help='Binarize the dataset?')
    parser.add_argument('--use_nn', action='store_true', default=False,
                        help='Use the larger model?')
    parser.add_argument('--dataset', choices=['MNIST', 'EMNIST'], default='MNIST', help='Choose a dataset')
    parser.add_argument('--model', type=str, metavar='model'
                        , help='Location of model')
    parser.add_argument('--class-index', type=int, metavar='C',
                        help='provide the class to train the model')
    parser.add_argument('--not-class', action='store_true', default=False,
                        help='Which class of the sigmoid to test for - if given test for the not class (not 9) else for class (9)')
    parser.add_argument('--func_f', choices=['min_distance', 'min_distance_plus_grid'], default='min_distance', help='Choose a dataset')
    args = parser.parse_args()
    pprint(args)
    ilp_solver(args, None)
