from __future__ import print_function

import argparse
import copy
import math
import os
import sys
import time
from pprint import pprint
import torch
import sys
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models import DualModel
import wandb
from loguru import logger
from torch.utils.data import Dataset, default_collate, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image

parent_dir = os.path.join(os.path.dirname(os.getcwd()))

# Append the parent directory path to system path
sys.path.append(parent_dir)
from load_dataset import load_dataset
from get_func_value import PGMLoss
from utils import create_directory, get_date_as_string

# Get the parent directory path
from model import LR, NeuralNetwork
from dataloader import get_pgm_dataset

def loss_function_from_func(X, y, func, args, device):
    g_x_y_tensor = func(X.to(device), y.to(device))
    return g_x_y_tensor


def print_range(tensor):
    # Find the minimum and maximum values in the tensor
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    # Print the range of the tensor
    print(f"Range of tensor: [{min_val}, {max_val}]")

# def get_total_loss(args, loss_value_from_g, loss_value_from_f, rho=10.0):

def get_total_loss(args, loss_value_from_g, loss_value_from_f, dual_outputs, rho):
    satisfied_loss = loss_value_from_f
    loss_value_from_g = torch.nn.functional.relu(loss_value_from_g.clone())
    loss_value_from_g = torch.squeeze(loss_value_from_g)
    dual_outputs = torch.squeeze(dual_outputs)
    if torch.any(torch.lt(dual_outputs, 0)):
        print("dual_outputs is less than 0")
        exit()
    g_x_loss_for_satisfied = torch.mul(loss_value_from_g.clone(), dual_outputs)
    satisfied_loss += g_x_loss_for_satisfied
    penalty = torch.pow(loss_value_from_g.clone(), 2) * (rho/2)
    non_satisfied_loss = satisfied_loss + penalty
    return satisfied_loss, non_satisfied_loss

def train_primal(primal, dual, data, train_loader,  func_g, func_f, optimizer_primal, rho, batch_idx, epoch, args, device):
    optimizer_primal.zero_grad()
    output = primal(data)
    output = torch.sigmoid(output)
    dual_outputs = dual(data)
    loss_value_from_g = loss_function_from_func(data, output, func_g, args, device)
    loss_value_from_f = loss_function_from_func(data, output, func_f, args, device)
    satisfied_loss, non_satisfied_loss = get_total_loss(
        args, loss_value_from_g, loss_value_from_f, dual_outputs, rho)
    val_g_y = loss_value_from_g.clone().detach()
    mask = (val_g_y > 0).float()
    mask = torch.squeeze(mask)
    result = torch.ones_like(satisfied_loss) - 2*mask
    loss = torch.where(result == 1, satisfied_loss, non_satisfied_loss,)
    loss = loss.mean()
    loss.backward()
    nn.utils.clip_grad_norm_(primal.parameters(), max_norm=5.0)
    optimizer_primal.step()
    final_loss = loss
    if batch_idx % args.log_interval == 0:
        logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\t Total Loss: {:.6f}  \t Satisfied Loss: {:.6f} \t Non Satisfied Loss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), final_loss.item(), satisfied_loss.mean().item(), (non_satisfied_loss).mean().item()))
        logger.info(f"Total number of constraints: {mask.shape[0]}")
        logger.info(f"Total number of non satisfied constraints: {torch.sum(mask)}")
    return loss.item()

def train_dual(primal, dual, copy_of_dual, data, func_g, func_f, optimizer_dual, rho, args, batch_idx, device):
    l2_loss = nn.MSELoss()
    optimizer_dual.zero_grad()
    primal_output = primal(data)
    primal_output = torch.sigmoid(primal_output)
    loss_value_from_g = loss_function_from_func(data, primal_output, func_g, args, device).float()
    loss_value_from_f = loss_function_from_func(data, primal_output, func_f, args, device)
    dual_outputs_frozen = copy_of_dual(data)
    current_iter_dual_outputs = dual(data)
    non_satisfied_output = loss_value_from_g
    first_tensor = dual_outputs_frozen + rho * non_satisfied_output
    second_tensor = torch.zeros_like(first_tensor)
    second_tensor[first_tensor > 0] = first_tensor[first_tensor > 0]
    loss = l2_loss(current_iter_dual_outputs, second_tensor)
    loss = loss.mean().float()
    loss.backward()
    nn.utils.clip_grad_norm_(dual.parameters(), max_norm=5.0)
    optimizer_dual.step()
    if batch_idx % args.log_interval == 0:
            logger.info('Dual Train :\t Loss: {:.6f} '.format(loss.item()))
            

def train(args, primal, dual, device, train_loader, optimizer_primal, optimizer_dual, func_f, func_g, epoch):
    primal.train()
    dual.train()
    rho = args.rho
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, _ = data.to(device), target.to(device),
        # Train Primal Model
        loss = train_primal(primal, dual, data, train_loader, func_g, func_f, optimizer_primal, rho, batch_idx, epoch, args, device)
        train_loss += loss
    copy_of_dual = copy.deepcopy(dual)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # Train Dual
        data, _ = data.to(device), target.to(device),
        train_dual(primal, dual, copy_of_dual, data, func_g, func_f, optimizer_dual,rho, args, batch_idx, device)
    train_loss /= len(train_loader.dataset)
    logger.info('====> Epoch: {} Average Train loss: {:.4f}'.format(epoch, train_loss))
    return train_loss 

def validate(args, model, func_f, device, test_loader, best_loss, counter):
    model.eval()
    test_loss = 0
    min_delta = 0.0005  # Minimum change in the validation loss to be considered as improvement
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            y = model(data)
            y = torch.sigmoid(y)
            loss_value_from_f = loss_function_from_func(data, y, func_f, args, device)
            loss_value_from_f = torch.divide(loss_value_from_f, args.num_var)
            loss = loss_value_from_f.mean()
            test_loss += loss.item()
    logger.info('\nTest set: Average loss: {:.4f}'.format(test_loss))
    if test_loss < best_loss - min_delta:
        best_loss = test_loss
        counter = 0
    else:
        counter += 1
    return best_loss, test_loss,  counter

def generate_output_images(model, test_loader, device):
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
    data_folder_path = args.data_path  # Replace with the path to the data_npz folder
    if not debug:
        wandb.init(project=f"ssl_PDL_{args.dataset}")
        wandb.config.update(args)
    # Define the output directory
    name = f"{args.dataset}"
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
        train_kwargs.update(cuda_kwargs)
        train_kwargs.update({'shuffle': True})
        test_kwargs.update(cuda_kwargs)
        test_kwargs.update({'shuffle': False})
    # Load the LR Model

    dataset = load_dataset(f"{data_folder_path}/{args.dataset}.npz")
    logger.info(f"You have selected {args.dataset}")
    train_data_set_from_ILP, test_data_set_from_ILP, X_boolean, num_X, num_Y = get_pgm_dataset(dataset)
    args.X_boolean = X_boolean
    train_loader = torch.utils.data.DataLoader(train_data_set_from_ILP, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_data_set_from_ILP, **train_kwargs)
    args.num_var = len(X_boolean)
    # Get F and G functions 
    f_univariate_functions, f_bivariate_functions = dataset['f_univariate_functions'], dataset['f_bivariate_functions']
    func_f = PGMLoss(f_univariate_functions, f_bivariate_functions, device)
    
    g_univariate_functions, g_bivariate_functions = dataset['g_univariate_functions'], dataset['g_bivariate_functions']
    func_g = PGMLoss(g_univariate_functions, g_bivariate_functions, device)
    # extract the number of features
    logger.info(f"Input Size - {num_X}, Output Size {num_Y}")
    hidden_size = [128, 256, 512]
    # Load the dataset
    # Initialize the autoencoder model for adversarial training
    primal = NeuralNetwork(num_X, hidden_size, num_Y).to(device)
    dual = DualModel(num_X,).to(device)
    optimizer_primal = optim.Adam(primal.parameters(), lr=args.lr)
    optimizer_dual = optim.Adam(dual.parameters(), lr=args.lr)
    # Define the learning rate scheduler
    lr_scheduler_primal = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_primal, mode='min', factor=args.gamma, threshold=1e-3, patience=2)
    lr_scheduler_dual = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_dual, mode='min', factor=args.gamma, threshold=1e-3, patience=2)
    best_loss = float('inf')
    counter = 0
    patience = 15  # Number of epochs to wait for the validation loss to improve
    # Train the model
    for epoch in range(1, args.ae_epochs + 1):
        train_loss = train(args, primal, dual, device,
              train_loader, optimizer_primal, optimizer_dual, func_f, func_g, epoch)
        best_loss, test_loss, counter = validate(
            args, primal, func_f, device, test_loader, best_loss, counter)
        lr_scheduler_primal.step(train_loss)
        lr_scheduler_dual.step(train_loss)
        print("Learning rate:", optimizer_primal.param_groups[0]['lr'])
        # if counter >= patience:
        #     print("Validation loss hasn't improved for {} epochs, stopping training...".format(patience))
        #     break
    
    # Save Actual and Adversarial Images
    all_inputs, all_outputs = generate_output_images(primal, test_loader, device)
    all_inputs = torch.tensor(all_inputs).float().to(device)
    all_outputs = torch.tensor(all_outputs).float().to(device)
    f_x_y_val = loss_function_from_func(all_inputs, all_outputs, func_f, args, device)
    f_x_y_val = torch.mean(f_x_y_val).item()

    g_x_y_val = loss_function_from_func(all_inputs, all_outputs, func_g, args, device)
    g_x_y_val = torch.where(g_x_y_val >= 0, torch.ones_like(g_x_y_val), torch.zeros_like(g_x_y_val))
    
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
        # np.savez(f"{model_outputs_dir}/test_outputs.npz", updated=all_outputs, initial=all_inputs)        
        if not os.path.exists(output_dir):
            # If the folder does not exist, create it
            os.makedirs(output_dir)
            print("Folder created successfully!")
        else:
            print("Folder already exists!")
        torch.save(primal.state_dict(), f'{output_dir}/primal.pt')
        torch.save(dual.state_dict(), f'{output_dir}/dual.pt')
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
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma'   , type=float, default=0.90, metavar='M',
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
    parser.add_argument('--rho', type=float, default=10, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--data-path', type=str, default="../datasets_npz/", metavar='DL',
                        help='Location of the datasets npz files')
    
    args = parser.parse_args()
    pprint(args)
    train_ae(args)
