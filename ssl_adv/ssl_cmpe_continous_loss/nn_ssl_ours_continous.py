from __future__ import print_function

import argparse
import copy
import math
import os
import sys
import time
from pprint import pprint
from func_f import f_xy_FunctionEvaluationModel

import torch
import sys
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from autoencoder import Autoencoder
from loguru import logger
from MNIST_dataloader import get_MNIST_data_loaders
from model_feasible_sol import FeasibleNet
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, Module
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, Subset, default_collate
from torchvision import datasets, transforms
from torchvision.utils import save_image

# Get the parent directory path
parent_dir = os.path.join(os.path.dirname(os.getcwd()), "train_classifier")

# Append the parent directory path to system path
sys.path.append(parent_dir)

from utils import create_directory, get_date_as_string

import wandb


# Get the parent directory path
parent_dir = os.path.join(os.path.dirname(os.getcwd()), "train_classifier")

# Append the parent directory path to system path
sys.path.append(parent_dir)

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


class LossFromK(nn.Module):
    def __init__(self):
        super(LossFromK, self).__init__()

    def forward(self, output, tmp_targets):
        loss = output
        return loss


def loss_from_g_alpha(y, loss_g, n_theta, args, device):
    if args.binarize:
        threshold = 0.5
        y = torch.where(y > threshold, torch.tensor(1.0).to(device), torch.tensor(0.0).to(device))

    logits = n_theta(y)
    outputs = torch.sigmoid(logits)
    tmp_target = torch.randn(1, 1).to(device)
    loss = loss_g(outputs, tmp_target)
    return loss, outputs, logits

def print_range(tensor):
    # Find the minimum and maximum values in the tensor
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)

    # Print the range of the tensor
    logger.info(f"Range of tensor: [{min_val}, {max_val}]")


def get_total_loss(args, data, output, loss_from_g, alpha_val, f_y, current_f_value, unit_g = False):
    # satisfied_loss = torch.mean(f_y(data, output), dim=1)
    satisfied_loss = current_f_value
    
    # Feasible value 
    if not args.not_class:
        # class 
        non_satisfied_loss = torch.relu((loss_from_g - 0.5))
    else:
        # not class
        non_satisfied_loss = torch.relu(-(loss_from_g - 0.5))
    non_satisfied_loss = torch.squeeze(non_satisfied_loss.clone())
    non_satisfied_loss += satisfied_loss
    non_satisfied_loss *= alpha_val
    if unit_g:
        non_satisfied_loss = torch.div(non_satisfied_loss, torch.abs(non_satisfied_loss.detach().clone()))
        non_satisfied_loss = non_satisfied_loss * 0.08
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
    # Create a mask where not_satisfied_mask is 0 and num_updates_for_bounds is greater than 0
    not_satisfied_mask = not_satisfied_mask.to(args.device)
    update_mask = (not_satisfied_mask == 0) & (num_updates_for_bounds > 1)
    # Update the previous_bounds based on the conditions
    # Update bounds if we have updated the bounds at least once
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
    # Update the previous_bounds with the current_g_value where not_satisfied_mask is 0 and num_updates_for_bounds is 0
    # Update the bounds if constraint are satisfied and we have not updated them yet
    previous_bounds = torch.where((not_satisfied_mask == 0) & (num_updates_for_bounds == 1), current_f_value, previous_bounds)
    # Update the num_updates_for_bounds where previous_bounds is updated
    num_updates_for_bounds = torch.where(update_mask, num_updates_for_bounds + 1, num_updates_for_bounds)
    num_updates_for_bounds = torch.where((not_satisfied_mask == 0) & (num_updates_for_bounds == 1), num_updates_for_bounds + 1, previous_bounds)
    return previous_bounds.to(args.device), num_updates_for_bounds.to(args.device)

def train(args, model, n_theta, device, train_dataset, optimizer, loss_ae, loss_g, epoch, train_kwargs):
    model.train()
    n_theta.eval()
    f_y = loss_ae
    train_dataloader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    total_loss = 0
    for batch_idx, (batch_data) in enumerate(train_dataloader):
        data, target, num_min, num_max, num_sum, num_updates_for_bounds, denom, idx = batch_data
        data, _ , num_min, num_max, num_sum, num_updates_for_bounds, denom = data.to(device), target.to(device), num_min.to(device), num_max.to(device), num_sum.to(device), num_updates_for_bounds.to(device), denom.to(device)
        data = data.reshape(-1, 28 * 28).to(device)
        # Select which operation to choose for the numerator
        optimizer.zero_grad()
        output = model(data)
        output = torch.sigmoid(output)
        loss_from_g, outputs_of_g, logits_of_g = loss_from_g_alpha(output, loss_g, n_theta, args, device)
        # Target is the same as the input since we want squared loss between input and output
        val_g_y = outputs_of_g.clone().detach()
        current_f_value = torch.mean(f_y(data, output), dim=1)
        # When the constraint is satisfied mask should be zero
        if not args.not_class:
            # For class (e.g. 9) case
            mask = (val_g_y > 0.5).float()
        else:
            # For not class (e.g. not 9) case
            mask = (val_g_y < 0.5).float()
        mask = torch.squeeze(mask)
        with torch.no_grad():
            # We are finding bounds so no need to allow gradient
            if args.numerator_operation == "min":
                updated_num_min, num_updates_for_bounds = update_bounds(not_satisfied_mask=mask, num_updates_for_bounds=num_updates_for_bounds, previous_bounds=num_min, current_f_value=current_f_value, numerator_operation=args.numerator_operation, args=args)
                alpha_val = torch.divide(updated_num_min, denom)
                train_dataset.update_bounds(new_bounds=updated_num_min, num_updates=num_updates_for_bounds, operation=args.numerator_operation, index=idx)
                # Check if we are updating the bounds in the dataset
                # assert torch.any((train_dataset.num_min[idx] != num_min)), "Not Updating the bounds in the dataset"
            elif args.numerator_operation == "max":
                updated_num_max, num_updates_for_bounds = update_bounds(not_satisfied_mask=mask, num_updates_for_bounds=num_updates_for_bounds, previous_bounds=num_max, current_f_value=current_f_value, numerator_operation=args.numerator_operation, args=args)
                alpha_val = torch.divide(updated_num_max, denom)
                train_dataset.update_bounds(new_bounds=updated_num_max, num_updates=num_updates_for_bounds, operation=args.numerator_operation, index=idx)
            elif args.numerator_operation == "average":
                updated_num_sum, num_updates_for_bounds = update_bounds(not_satisfied_mask=mask, num_updates_for_bounds=num_updates_for_bounds, previous_bounds=num_sum, current_f_value=current_f_value, numerator_operation=args.numerator_operation, args=args)
                alpha_val = torch.divide(torch.divide(updated_num_sum, num_updates_for_bounds), denom)
                train_dataset.update_bounds(new_bounds=updated_num_sum, num_updates=num_updates_for_bounds, operation=args.numerator_operation, index=idx)
            else:
                raise ValueError("Invalid numerator operation")
        # We can multiply alpha with any positive constant. Makes convergence faster
        alpha_val = alpha_val * 20
        assert torch.all(alpha_val >= 0), "Alpha value should be greater than 0"
        satisfied_loss, non_satisfied_loss = get_total_loss(args, data, output, loss_from_g, alpha_val, f_y, current_f_value)
        # Continous Loss 
        logits_of_g = torch.squeeze(logits_of_g).detach().clone()
        logit_for_loss = args.beta*logits_of_g
        sigmoid_for_loss = torch.sigmoid(logit_for_loss)
        loss = (1 - sigmoid_for_loss) * satisfied_loss + sigmoid_for_loss * non_satisfied_loss
        loss = loss.mean()
        if not args.no_entropy_loss:
            # entropy_loss = -(output * torch.log(output)).sum(dim=1).mean()
            entropy_loss = entropy_loss_function(output.clone())
            
            loss += args.entropy_weight * entropy_loss
        
        total_loss += loss.item()
        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        final_loss = loss
        wandb.log({f"train_loss": final_loss.item()}, )
        if batch_idx % args.log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\t Total Loss: {:.6f} '.format(
            epoch, batch_idx * len(data), len(train_dataset),
            100. * batch_idx / len(train_dataset), final_loss.item()))
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

def validate(args, model, n_theta, device, test_dataset, f_xy, best_loss, counter, test_kwargs):
    model.eval()
    n_theta.eval()
    test_loss = 0
    correct = 0
    correct_2 = 0
    min_delta = 0.0005  # Minimum change in the validation loss to be considered as improvement
    test_dataloader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
    
    with torch.no_grad():
        for batch_data in test_dataloader:
            data, target, num_min, num_max, num_sum, num_updates_for_bounds, denom, idx = batch_data
            data, target = data.to(device), target.to(device)
            data = data.reshape(-1, 28 * 28).to(device)
            y = model(data)
            y = torch.sigmoid(y)
            loss = f_xy(data, y).mean()
            test_loss += loss.item()
            if args.binarize:
                binarized_y = (y > 0.5).float()
                output = n_theta(binarized_y)
            else:
                output = n_theta(y)
            output = torch.sigmoid(output)
            pred = (output > 0.5).float()
            correct += pred.eq(target.view_as(pred)).sum().item()
            output = n_theta(data)
            output = torch.sigmoid(output)
            pred = (output > 0.5).float()
            correct_2 += pred.eq(target.view_as(pred)).sum().item()

    wandb.log({f"test_loss_{args.beta}": test_loss}, )
    wandb.log({"test_accuracy for adv example": 100. * correct / len(test_dataloader.dataset)}, )
    logger.info(
        '\nTest set: Average loss: {:.4f}, Adv. Accuracy: {}/{} ({:.0f}%, Actual Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_dataloader.dataset),
            100. * correct /
            len(test_dataloader.dataset), correct_2, len(test_dataloader.dataset),
            100. * correct_2 / len(test_dataloader.dataset)))
    if test_loss < best_loss - min_delta:
        best_loss = test_loss
        counter = 0
    else:
        counter += 1
    # correct - for examples generated adversarially
    adv_accuracy = 100. * correct / len(test_dataloader.dataset)
    true_accuracy = 100. * correct_2 / len(test_dataloader.dataset)
    # correct_2 - Input example images
    return best_loss, test_loss,  counter, adv_accuracy, true_accuracy


def generate_output_images(model, test_dataset, device, output_dir, test_kwargs):
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
    with torch.no_grad():
        all_inputs = []
        all_outputs = []
        test_dataloader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        for batch_data in test_dataloader:
            data, target, num_min, num_max, num_sum, num_updates_for_bounds, denom, idx = batch_data
            data, target = data.to(device), target.to(device)
            data = data.reshape(-1, 28 * 28).to(device)
            y = model(data)
            y = torch.sigmoid(y)
            if args.binarize:
                y = (y > 0.5).float()
            # Save the output images
            all_outputs.extend(y.detach().cpu().tolist())
            all_inputs.extend(data.detach().cpu().tolist())
            y = y.reshape(-1, 28, 28).to(device)
            data = data.reshape(-1, 28, 28).to(device)
            for idx in range(data.shape[0]):
                concatenated_image = torch.cat(
                    (data[idx].detach(), y[idx].detach()), dim=1)
                save_image(concatenated_image,
                           f'{output_dir}/img_{save_idx}.png')
                save_idx += 1
    return all_inputs, all_outputs

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
        nn_as_dict_of_np_array[f"{param_tensor}"] = nn.state_dict()[
            param_tensor].cpu().detach().numpy()
    return nn_as_dict_of_np_array, nn


def train_ae(args, model=None):
    debug = False
    args.debug = debug
    if debug:
        
        args.ae_epochs = 10
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()
    import sys
    class_label = int(not args.not_class)
    select_class = args.class_index
    if not debug and model is None:
        wandb.init(project=f"ssl_continous_no_lag_{args.numerator_operation}_{select_class}_{class_label}_beta_{args.beta}")
        wandb.config.update(args)
    # Define the output directory
    if not args.not_class:
        name = f"{args.class_index}/{args.numerator_operation}_{args.class_index}_beta_{args.beta}"
    else:
        name = f"{args.class_index}/not_{args.numerator_operation}_{args.class_index}_beta_{args.beta}"
    if args.binarize:
        output_dir = f"models/binarized/{name}"
    else:
        output_dir = f"models/continous/{name}"
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
    # Load the classifier Model
    if model is None:
        if not args.use_nn:
            n_theta_as_dict_of_np_array, n_theta = load_nn(args.model,
                                                           use_cuda)

        # Load the large model
        else:
            n_theta_as_dict_of_np_array, n_theta = load_nn(
                args.model, use_cuda)
    else:
        n_theta = model
    # Freeze all parameters of the n_theta network
    for param in n_theta.parameters():
        param.requires_grad = False
    
    # Define the function f(x,y)
    negative_weight = 1e-2
    positive_weight = 2.0
    factor_f = torch.tensor([negative_weight, positive_weight , positive_weight, negative_weight]).to(device)
    f_xy = nn.DataParallel(f_xy_FunctionEvaluationModel(factor_f))
    logger.info(f"You have selected {select_class}")
    logger.info(f"You have selected {class_label}")

    if args.dataset.lower() == 'mnist':
        datasets = get_MNIST_data_loaders(args, n_theta, n_theta_as_dict_of_np_array, f_xy, train_kwargs, test_kwargs,
                                                                                     select_class=select_class, class_label=class_label, device=device)
    if not args.not_class:
        dataset_for_1 = datasets['1']
        train_dataset, test_dataset = dataset_for_1
    else:
        dataset_for_0 = datasets['0']
        train_dataset, test_dataset= dataset_for_0

    input_size = 28 * 28
    # Load the dataset
    # Initialize the autoencoder model for adversarial training
    autoencoder = Autoencoder(input_size,).to(device)
    # Define the loss function and optimizer
    optimizer = optim.Adam(autoencoder.parameters(), lr=args.lr)
    # Define the learning rate scheduler
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.gamma,  patience=10)
    loss_ae = f_xy
    loss_g = LossFromK()
    best_loss = float('inf')
    counter = 0
    patience = 15  # Number of epochs to wait for the validation loss to improve
    # Train the model
    
    for epoch in range(1, args.ae_epochs + 1):
        train_loss = train(args, autoencoder, n_theta, device,
              train_dataset, optimizer, loss_ae, loss_g, epoch, train_kwargs)
        best_loss, test_loss, counter, adv_accuracy, true_accuracy = validate(
            args, autoencoder, n_theta, device, test_dataset, f_xy, best_loss, counter, test_kwargs)
        print("Learning rate:", optimizer.param_groups[0]['lr'])
        # if counter >= patience:
        #     print("Validation loss hasn't improved for {} epochs, stopping training...".format(patience))
        #     break
        lr_scheduler.step(train_loss)
    
    

    # Save Actual and Adversarial Images
    if args.binarize:
        all_inputs, all_outputs = generate_output_images(autoencoder, test_dataset, device, output_dir.replace("models", "adv_images_binarized"), test_kwargs)
    else:
        all_inputs, all_outputs = generate_output_images(autoencoder, test_dataset, device, output_dir.replace("models", "adv_images_continous"), test_kwargs)
    # Save the model
    if args.save_model:
        model_outputs_dir = output_dir.replace("models", "model_outputs")
        if not os.path.exists(model_outputs_dir):
        # If the folder does not exist, create it
            os.makedirs(model_outputs_dir)
            print("Folder created successfully!")
        else:
            print("Folder already exists!")
        np.savez(f"{model_outputs_dir}/test_outputs.npz", updated=all_outputs, initial=all_inputs)        
        if not os.path.exists(output_dir):
            # If the folder does not exist, create it
            os.makedirs(output_dir)
            print("Folder created successfully!")
        else:
            print("Folder already exists!")
        torch.save(autoencoder.state_dict(),
                   f'{output_dir}/adv_example_generator.pt')


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
    parser.add_argument(
        '--dataset', choices=['MNIST', 'EMNIST'], default='MNIST', help='Choose a dataset')
    parser.add_argument(
        '--numerator_operation', choices=['min', 'max', 'average'], default='min', help='Choose the numerator operation')
    parser.add_argument('--model', type=str, metavar='model',
                        help='Location of model')
    parser.add_argument('--class-index', type=int, metavar='C',
                        help='provide the class to train the model')
    parser.add_argument('--not-class', action='store_true', default=False,
                        help='Which class of the sigmoid to test for - if given test for the not class (not 9) else for class (9)')
    parser.add_argument('--entropy_weight', type=float, default=1e-2, metavar='LR',
                        help='entropy_weight (default: 0.001)')
    parser.add_argument('--no-entropy-loss', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--penalty', action='store_true', default=False,
                        help='Adds penalty term to train')
    parser.add_argument('--beta', type=float, default=5, metavar='LR',
                        help='Beta Value (default: 1.0)')

    args = parser.parse_args()
    pprint(args)
    train_ae(args, None)
