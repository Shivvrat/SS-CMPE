from __future__ import print_function
import argparse
import copy
import csv
import glob
import os
import math
from pprint import pprint
import time

import numpy as np
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

from utils import create_directory, get_date_as_string

parent_dir = os.path.join(os.path.dirname(os.getcwd()))

# Append the parent directory path to system path
sys.path.append(parent_dir)
from load_dataset import load_dataset
from get_func_value import PGMLoss


def print_range(tensor):
    # Find the minimum and maximum values in the tensor
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)

    # Print the range of the tensor
    print(f"Range of tensor: [{min_val}, {max_val}]")

def loss_function_from_func(X, y, func, args, device):
    g_x_y_tensor = func(X.to(device), y.to(device))
    return g_x_y_tensor

def ilp_solver(args):
    debug = False
    args.debug = debug
    if not debug:
        wandb.init(project=f"pgm_ilp_{args.dataset}")
        wandb.config.update(args)
    output_dir = f"models/{args.dataset}"
    # Add a logger to the project
    config = {
        "handlers": [
            {"sink": sys.stdout,
             "format": "<green>{time:YYYY-MM-DD at HH:mm:ss}</green> | {module}.{function} | <level>{message}</level> "},
            {"sink": f"{output_dir}/" + "logger_{time}.log",
             "format": "<green>{time:YYYY-MM-DD at HH:mm:ss}</green> | {module}.{function} | <level>{message}</level> "}
        ],
        "extra": {"user": "sva"}
    }
    logger.configure(**config)
    # torch.manual_seed(args.seed)
    folder_path = args.data_path  # Replace with the path to the data_npz folder

    # Get all files in the folder using glob
    files = glob.glob(folder_path + "/*")
    f_x_y_values = []
    datasets = []
    # Print the list of files
    for file in files:
        dataset_name = os.path.basename(file)
        datasets.append(dataset_name)
        dataset = load_dataset(file)
        X = torch.tensor(dataset['X_test']).float()
        y = torch.tensor(dataset['samples_optimal_y_test']).float()
        X_boolean = torch.tensor(dataset['X_boolean']).bool()
        assert X.shape[1] + y.shape[1] == X_boolean.shape[0]
        logger.info(f"Dataset Name {dataset_name}")
        logger.info(f"Number of X {torch.sum(X_boolean)}")
        logger.info(f"Number of y {X_boolean.shape[0] - torch.sum(X_boolean)}")
        f_univariate_functions, f_bivariate_functions = dataset['f_univariate_functions'], dataset['f_bivariate_functions']
        g_univariate_functions, g_bivariate_functions = dataset['g_univariate_functions'], dataset['g_bivariate_functions']
        logger.info(f"Number of f univariate functions {len(f_univariate_functions)}")
        logger.info(f"Number of f bivariate functions {len(f_bivariate_functions)}")
        logger.info(f"Number of g univariate functions {len(g_univariate_functions)}")
        logger.info(f"Number of g bivariate functions {len(g_bivariate_functions)}")
        device = torch.device('cuda')
        func_f = PGMLoss(f_univariate_functions, f_bivariate_functions, device=torch.device('cuda'))
        func_g = PGMLoss(g_univariate_functions, g_bivariate_functions, device=torch.device('cuda'))
        f_x_y_val = loss_function_from_func(X, y, func_f, args, device)
        f_x_y_val = torch.mean(f_x_y_val).item()
        f_x_y_values.append(f_x_y_val)
        g_x_y_val = loss_function_from_func(X, y, func_g, args, device)
        logger.info(f"Range of G = {torch.min(g_x_y_val)}, {torch.max(g_x_y_val)}")
        g_x_y_val = torch.where(g_x_y_val >= 1e-3, torch.ones_like(g_x_y_val), torch.zeros_like(g_x_y_val))
        logger.info(f"Objective value for dataset {dataset_name} = {f_x_y_val}")
        logger.info(f"Number of violations for dataset {dataset_name} = {torch.sum(g_x_y_val)}")
        wandb.log({f"f_x_y_val for dataset {dataset_name}": f_x_y_val})
        wandb.log({f"Number of violations for dataset {dataset_name}": torch.sum(g_x_y_val)})
    final_strings = [datasets, f_x_y_values]
    with open("objective_values.csv", "w", newline="") as file:
        writer = csv.writer(file)
        # Write each inner list as a separate row in the CSV file
        for row in final_strings:
            writer.writerow(row)

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
    parser.add_argument('--dataset', default='dataset-seg12-60', help='Choose a dataset')
    parser.add_argument('--data-path', type=str, metavar='DL',  help='Location of the datasets npz files')
    
    args = parser.parse_args()
    pprint(args)
    ilp_solver(args)
