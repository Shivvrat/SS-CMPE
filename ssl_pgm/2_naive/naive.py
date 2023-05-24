from __future__ import print_function
import argparse
import copy
import os
import math
from pprint import pprint
import time
import torch
import sys
import numpy as np
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

parent_dir = os.path.join(os.path.dirname(os.getcwd()))

# Append the parent directory path to system path
sys.path.append(parent_dir)
from load_dataset import load_dataset
from get_func_value import PGMLoss


from utils import create_directory, get_date_as_string

# Get the parent directory path
from model import NeuralNetwork
from dataloader import get_pgm_dataset

def train(args, model, device, train_loader, optimizer, loss_ae, epoch):
    model.train()
    f_y = loss_ae
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device), 
        optimizer.zero_grad()
        output = model(data)
        output = torch.sigmoid(output)
        loss = f_y(output, target)
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        final_loss = loss
        train_loss += final_loss.item()
        if batch_idx % args.log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\t Total Loss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), final_loss.item()))
            if args.dry_run:
                break
    train_loss /= len(train_loader.dataset)
    logger.info('\Train set: Average loss: {:.4f}'.format(train_loss))
                

def validate(args, model, device, test_loader, loss_function, best_loss, counter):
    model.eval()
    test_loss = 0
    min_delta = 0.001  # Minimum change in the validation loss to be considered as improvement
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            y = model(data)
            y = torch.sigmoid(y)
            loss = loss_function(y, target.float())
            loss = loss.mean()
            test_loss += loss.item()
    logger.info('\nTest set: Average loss: {:.4f}'.format(test_loss))
    if test_loss < best_loss - min_delta:
        best_loss = test_loss
        counter = 0
    else:
        counter += 1
    return best_loss, counter

def generate_outputs(model, test_loader, device):
    logger.info("Generating outputs")
    with torch.no_grad():
        all_inputs = []
        all_outputs = []
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            y = model(data)
            y = torch.sigmoid(y)
            # Save the outputs
            all_outputs.extend(y.detach().cpu().tolist())
            all_inputs.extend(data.detach().cpu().tolist())
    return all_inputs, all_outputs


def train_ae(args):
    debug = False
    args.debug = debug

    if debug:
        
        args.ae_epochs = 10
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()
    import sys
    data_folder_path = args.data_path # Replace with the path to the data_npz folder

    name = f"{args.dataset}_{args.loss}"

    if not debug:
        wandb.init(project=f"sl_naive_{name}")
        wandb.config.update(args)
    # Define the output directory
    output_dir = f"models/{name}"
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
    if use_cuda:
        logger.info("Using GPU for training")
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 4,
                       'pin_memory': True,
                       }
        train_kwargs |= cuda_kwargs
        train_kwargs.update({'shuffle': True})
        test_kwargs |= cuda_kwargs
        test_kwargs.update({'shuffle': False})
    dataset = load_dataset(f"{data_folder_path}/{args.dataset}.npz")
    logger.info(f"You have selected {args.dataset}")
    train_data_set_from_ILP, test_data_set_from_ILP, X_boolean, num_X, num_Y = get_pgm_dataset(dataset)
    
    train_loader_for_model = torch.utils.data.DataLoader(train_data_set_from_ILP, **train_kwargs)
    test_loader_for_model = torch.utils.data.DataLoader(test_data_set_from_ILP, **train_kwargs)
    
    # Get F and G functions 
    f_univariate_functions, f_bivariate_functions = dataset['f_univariate_functions'], dataset['f_bivariate_functions']
    func_f = PGMLoss(f_univariate_functions, f_bivariate_functions, device)
    
    
    g_univariate_functions, g_bivariate_functions = dataset['g_univariate_functions'], dataset['g_bivariate_functions']
    func_g = PGMLoss(g_univariate_functions, g_bivariate_functions, device)
    
    # extract the number of features
    logger.info(f"Input Size - {num_X}, Output Size {num_Y}")
    hidden_size = [128, 256, 512]
    # Load the dataset
    # Initialize the model model for training
    model = NeuralNetwork(num_X, hidden_size, num_Y).to(device)
    # Define the loss function and optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.loss == "MSE":
        loss_ae = nn.MSELoss()
    elif args.loss == "MAE":
        loss_ae = nn.L1Loss()
    else:
        logger.log("Invalid loss function")
    best_loss = float('inf')
    counter = 0
    patience = 10  # Number of epochs to wait for the validation loss to improve
    # Train the model
    for epoch in range(1, args.ae_epochs + 1):
        train(args, model, device, train_loader_for_model, optimizer, loss_ae, epoch)
        _, _  = validate(args, model, device, train_loader_for_model, loss_ae,  best_loss, counter)
        best_loss, counter = validate(args, model, device, test_loader_for_model, loss_ae, best_loss, counter)
        # if counter >= patience:
        #     print("Validation loss hasn't improved for {} epochs, stopping training...".format(patience))
        #     break
    # Save Actual and Adversarial Images
    all_inputs, all_outputs = generate_outputs(model, test_loader_for_model, device)
    all_inputs = torch.tensor(all_inputs).float()
    all_outputs = torch.tensor(all_outputs).float()
    f_x_y_tensor = func_f(all_inputs.to(device), all_outputs.to(device))
    # Take mean over all examples
    f_x_y_val = torch.mean(f_x_y_tensor).item()
    g_x_y_tensor = func_g(all_inputs.to(device), all_outputs.to(device))
    g_x_y_val = torch.where(g_x_y_tensor >= 0, torch.ones_like(g_x_y_tensor), torch.zeros_like(g_x_y_tensor))
    # Save the model
    if args.save_model:
        model_outputs_dir = output_dir.replace("models", "model_outputs")
        if not os.path.exists(model_outputs_dir):
        # If the folder does not exist, create it
            os.makedirs(model_outputs_dir)
            print("Folder created successfully!")
        else:
            print("Folder already exists!")
        np.savez(f"{model_outputs_dir}/test_outputs.npz", updated=all_outputs.cpu().detach().numpy(), initial=all_inputs.cpu().detach().numpy(), objective=f_x_y_val, violations=torch.sum(g_x_y_val).item())       
        if not os.path.exists(output_dir):
        # If the folder does not exist, create it
            os.makedirs(output_dir)
            print("Folder created successfully!")
        else:
            print("Folder already exists!")
        torch.save(model.state_dict(), f'{output_dir}/adv_example_generator.pt')
    logger.info(f"Objective value for dataset {args.dataset} = {f_x_y_val}")
    logger.info(f"Number of violations for dataset {args.dataset} = {torch.sum(g_x_y_val)}")
    wandb.log({"f_x_y_val": f_x_y_val})
    wandb.log({f"Number of violations for dataset {args.dataset}": torch.sum(g_x_y_val)})    

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
    parser.add_argument('--dataset',  help='Choose a dataset')
    parser.add_argument('--loss', choices=['MSE', 'MAE'], default='MSE', help='Choose the loss function')
    parser.add_argument('--data-path', type=str, default="../datasets_npz/", metavar='DL',
                        help='Location of the datasets npz files')
    
    args = parser.parse_args()
    pprint(args)
    train_ae(args)
