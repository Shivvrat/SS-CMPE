from __future__ import print_function

import argparse
import copy
import math
import os
import sys
import time
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
from torchvision.utils import save_image

parent_dir = os.path.join(os.path.dirname(os.getcwd()))

# Append the parent directory path to system path
sys.path.append(parent_dir)
from load_dataset import load_dataset
from get_func_value import PGMLoss

from utils import create_directory, get_date_as_string

# Get the parent directory path
from model import LR, NeuralNetwork
from PGM_dataloader_ours import get_pgm_dataset

# Check gradient flow
# from utils import plot_grad_flow

def loss_function_from_func(X, y, func, args, device):
    g_x_y_tensor = func(X.to(device), y.to(device))
    return g_x_y_tensor


def print_range(tensor):
    # Find the minimum and maximum values in the tensor
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    # Print the range of the tensor
    print(f"Range of tensor: [{min_val}, {max_val}]")

def get_total_loss(args, loss_value_from_g, loss_value_from_f, alpha_val,):
    satisfied_loss = loss_value_from_f
    # Feasible value 
    non_satisfied_loss = torch.nn.functional.relu(loss_value_from_g.clone())
    non_satisfied_loss = torch.squeeze(non_satisfied_loss.clone())
    # base_non_satisfied_loss = non_satisfied_loss.clone()
    non_satisfied_loss += satisfied_loss
    if args.penalty:
        penalty = torch.pow(torch.squeeze(non_satisfied_loss.clone()), 2) * (10/2)
        non_satisfied_loss += penalty
    non_satisfied_loss *= alpha_val
    # non_satisfied_loss += torch.pow(base_non_satisfied_loss, 2)
    
    return satisfied_loss, non_satisfied_loss



def update_bounds(not_satisfied_mask, num_updates_for_bounds, previous_bounds, current_f_value, numerator_operation, args):
    """
    Updates the bounds based on given conditions.

    Args:
        not_satisfied_mask (torch.Tensor): A binary mask indicating whether conditions are satisfied (0) or not (1).
        num_updates_for_bounds (torch.Tensor): The number of updates made to the bounds.
        previous_bounds (torch.Tensor): The previous bounds to be updated.
        current_f_value (torch.Tensor): The current f value.

    Returns:
        torch.Tensor: Updated bounds for f.
        torch.Tensor: Updated number of updates for bounds.
    """
    not_satisfied_mask = not_satisfied_mask.to(args.device)
    update_mask = (not_satisfied_mask == 0) & (num_updates_for_bounds > 1)
    if numerator_operation=="max":
        # Select max value between previous_bounds and current_g_value
        previous_bounds = torch.where(update_mask,
                                    torch.where(previous_bounds > current_f_value, previous_bounds, current_f_value),
                                    previous_bounds)
    elif numerator_operation=="min":
        # Select min value between previous_bounds and current_g_value
        previous_bounds = torch.where(update_mask,
                                    torch.where(previous_bounds < current_f_value, previous_bounds, current_f_value),
                                    previous_bounds)
    elif numerator_operation=="average":
        # Add all the values of previous_bounds and current_g_value
        previous_bounds = torch.where(update_mask,
                                    previous_bounds + current_f_value,
                                    previous_bounds)
    previous_bounds = torch.where((not_satisfied_mask == 0) & (num_updates_for_bounds == 1), current_f_value, previous_bounds)
    # Update the num_updates_for_bounds where previous_bounds is updated
    num_updates_for_bounds = torch.where(update_mask, num_updates_for_bounds + 1, num_updates_for_bounds)
    num_updates_for_bounds = torch.where((not_satisfied_mask == 0) & (num_updates_for_bounds == 1), num_updates_for_bounds + 1, num_updates_for_bounds)
    return previous_bounds.to(args.device), num_updates_for_bounds.to(args.device)

def train(args, model, device, train_dataset, optimizer, func_f, func_g, epoch, train_kwargs):
    model.train()
    train_dataloader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    total_loss = 0
    for batch_idx, (batch_data) in enumerate(train_dataloader):
        data, target, num_min, num_max, num_sum, num_updates_for_bounds, denom, idx = batch_data
        data, _ , num_min, num_max, num_sum, num_updates_for_bounds, denom = data.to(device), target.to(device), num_min.to(device), num_max.to(device), num_sum.to(device), num_updates_for_bounds.to(device), denom.to(device)
        # Select which operation to choose for the numerator
        optimizer.zero_grad()
        
        output = model(data)
        output = torch.sigmoid(output)
        loss_value_from_g = loss_function_from_func(data, output, func_g, args, device)
        loss_value_from_f = loss_function_from_func(data, output, func_f, args, device)
        val_g_y = loss_value_from_g.clone().detach()
        mask = (val_g_y > 0).float()
        mask = torch.squeeze(mask)
        with torch.no_grad():
            # We are finding bounds so no need to allow gradient
            if args.numerator_operation == "min":
                updated_num_min, num_updates_for_bounds = update_bounds(not_satisfied_mask=mask, num_updates_for_bounds=num_updates_for_bounds, previous_bounds=num_min, current_f_value=loss_value_from_f, numerator_operation=args.numerator_operation, args=args)
                alpha_val = torch.divide(updated_num_min, denom)
                train_dataset.update_bounds(new_bounds=updated_num_min.float(), num_updates=num_updates_for_bounds, operation=args.numerator_operation, index=idx)
                # Check if we are updating the bounds in the dataset
                # assert torch.any((train_dataset.num_min[idx] != num_min)), "Not Updating the bounds in the dataset"
            elif args.numerator_operation == "max":
                updated_num_max, num_updates_for_bounds = update_bounds(not_satisfied_mask=mask, num_updates_for_bounds=num_updates_for_bounds, previous_bounds=num_max, current_f_value=loss_value_from_f, numerator_operation=args.numerator_operation, args=args)
                alpha_val = torch.divide(updated_num_max, denom)
                train_dataset.update_bounds(new_bounds=updated_num_max.float(), num_updates=num_updates_for_bounds, operation=args.numerator_operation, index=idx)
            elif args.numerator_operation == "average":
                updated_num_sum, num_updates_for_bounds = update_bounds(not_satisfied_mask=mask, num_updates_for_bounds=num_updates_for_bounds, previous_bounds=num_sum, current_f_value=loss_value_from_f, numerator_operation=args.numerator_operation, args=args)
                alpha_val = torch.divide(torch.divide(updated_num_sum, num_updates_for_bounds), denom)
                train_dataset.update_bounds(new_bounds=updated_num_sum.float(), num_updates=num_updates_for_bounds, operation=args.numerator_operation, index=idx)
            else:
                raise ValueError("Invalid numerator operation")
        # We can multiply alpha with any positive constant. Makes convergence faster
        alpha_val = alpha_val*args.alpha_multiplier
        satisfied_loss, non_satisfied_loss = get_total_loss(args, loss_value_from_g, loss_value_from_f, alpha_val)
        loss_g_clone = torch.squeeze(loss_value_from_g.clone().detach())
        loss_g_clone = args.beta*loss_g_clone
        sigmoid_for_loss = torch.sigmoid(loss_g_clone)
        loss = (1 - sigmoid_for_loss) * satisfied_loss + sigmoid_for_loss * non_satisfied_loss
        loss = loss.mean()
        if not args.no_entropy_loss:
            # entropy_loss = -(output * torch.log(output)).sum(dim=1).mean()
            entropy_loss = entropy_loss_function(output.clone())
            
            loss += args.entropy_weight * entropy_loss
        
        total_loss += loss.item()
        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        final_loss = loss
        if batch_idx % args.log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\t Total Loss: {:.6f}  \t Satisfied Loss: {:.6f} \t Non Satisfied Loss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_dataset),
            100. * batch_idx / len(train_dataset), final_loss.item(), satisfied_loss.mean().item(), (non_satisfied_loss).mean().item()))
            logger.info(f"Total number of constraints: {mask.shape[0]}")
            logger.info(f"Total number of non satisfied constraints: {torch.sum(mask)}")
            if args.dry_run:
                break
    total_loss /= len(train_dataset)
    logger.info(f"Total loss: {total_loss}")
    wandb.log({f"train_total_loss": total_loss}, )
    return total_loss            


# Example loss function using torch.log()
def entropy_loss_function(predictions, epsilon=1e-5):
    loss =  -(predictions * torch.log(predictions + epsilon))
    return loss.mean()

def validate(args, model, device, test_dataset, func_f, best_loss, counter, test_kwargs):
    model.eval()
    test_loss = 0
    min_delta = 0.0005  # Minimum change in the validation loss to be considered as improvement
    test_dataloader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
    with torch.no_grad():
        for batch_data in test_dataloader:
            data, target, num_min, num_max, num_sum, num_updates_for_bounds, denom, idx = batch_data
            data, target = data.to(device), target.to(device)
            y = model(data)
            y = torch.sigmoid(y)
            loss_value_from_f = loss_function_from_func(data, y, func_f, args, device)
            loss_value_from_f = torch.divide(loss_value_from_f, args.num_var)
            loss = loss_value_from_f.mean()
            test_loss += loss.item()
    wandb.log({"test_loss": test_loss}, )
    logger.info('\nTest set: Average loss: {:.4f}'.format(test_loss))
    if test_loss < best_loss - min_delta:
        best_loss = test_loss
        counter = 0
    else:
        counter += 1
    return best_loss, test_loss, counter

def generate_output_images(model, test_dataloader, device):
    logger.info("Generating outputs")
    with torch.no_grad():
        all_inputs = []
        all_outputs = []
        for batch_data in test_dataloader:
            data, target, num_min, num_max, num_sum, num_updates_for_bounds, denom, idx = batch_data
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
    name = f"{args.dataset}/{args.numerator_operation}/"
    if args.penalty:
        name = f"{args.dataset}/penalty_{args.numerator_operation}/"
    else:
        name = f"{args.dataset}/{args.numerator_operation}/"
    if not debug:
        wandb.init(project=f"ssl_ours_{args.dataset}_{args.numerator_operation}")
        wandb.config.update(args)
    # Define the output directory
    output_dir = f"models/{name}"
    # Add a logger to the project

    # torch.manual_seed(args.seed)
    if use_cuda:
        logger.info("Using GPU for training")
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    args.device = device
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 0,
                       'pin_memory': False,
                       }
        train_kwargs.update(cuda_kwargs)
        train_kwargs.update({'shuffle': True})
        test_kwargs.update(cuda_kwargs)
        test_kwargs.update({'shuffle': False})
    dataset = load_dataset(f"{data_folder_path}/{args.dataset}.npz")
    logger.info(f"You have selected {args.dataset}")
    train_dataset, test_dataset, X_boolean, num_X, num_Y = get_pgm_dataset(dataset, device)
    args.X_boolean = X_boolean
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **train_kwargs)
    args.num_var = len(X_boolean)
    # Get F and G functions 
    # Define Loss from a PGM 
    f_univariate_functions, f_bivariate_functions = dataset['f_univariate_functions'], dataset['f_bivariate_functions']
    func_f = PGMLoss(f_univariate_functions, f_bivariate_functions, device)
    g_univariate_functions, g_bivariate_functions = dataset['g_univariate_functions'], dataset['g_bivariate_functions']
    func_g = PGMLoss(g_univariate_functions, g_bivariate_functions, device)
    # extract the number of features
    logger.info(f"Input Size - {num_X}, Output Size {num_Y}")
    hidden_size = [128, 256, 512]
    # Initialize the model model for training
    model = NeuralNetwork(num_X, hidden_size, num_Y).to(device)
    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # Define the learning rate scheduler
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.gamma,  patience=10)
    best_loss = float('inf')
    counter = 0
    patience = 15  # Number of epochs to wait for the validation loss to improve
    # Train the model
    for epoch in range(1, args.ae_epochs + 1):
        train_loss = train(args, model, device,
              train_dataset, optimizer, func_f, func_g, epoch, train_kwargs)
        best_loss, test_loss, counter= validate(
            args, model, device, test_dataset, func_f, best_loss, counter, test_kwargs)
        lr_scheduler.step(train_loss)
        print("Learning rate:", optimizer.param_groups[0]['lr'])
        # if counter >= patience:
        #     print("Validation loss hasn't improved for {} epochs, stopping training...".format(patience))
        #     break

    # Save Actual and Adversarial Images
    all_inputs, all_outputs = generate_output_images(model, test_loader, device)
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
        if not os.path.exists(output_dir):
            # If the folder does not exist, create it
            os.makedirs(output_dir)
            print("Folder created successfully!")
        else:
            print("Folder already exists!")
        torch.save(model.state_dict(), f'{output_dir}/model.pt')
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
    parser.add_argument('--gamma', type=float, default=0.90, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--alpha_multiplier', type=float, default=25, metavar='LR',
                        help='We can multiply alpha with any positive constant. Makes convergence faster')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument(
        '--numerator_operation', choices=['min', 'max', 'average'], default='min', help='Choose the numerator operation')
    parser.add_argument('--entropy_weight', type=float, default=1e-2, metavar='LR',
                        help='entropy_weight (default: 0.001)')
    parser.add_argument('--no-entropy-loss', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dataset', help='Choose a dataset')
    parser.add_argument('--penalty', action='store_true', default=False,
                        help='Adds penalty term to train')
    parser.add_argument('--beta', type=float, default=2, metavar='LR',
                        help='Beta Value (default: 1.0)')
    parser.add_argument('--data-path', type=str, metavar='DL',  help='Location of the datasets npz files')
    
    args = parser.parse_args()
    pprint(args)
    train_ae(args)
