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
from models import Autoencoder, DualModel
from loguru import logger
from MNIST_dataloader import get_MNIST_data_loaders_test
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
    return loss, outputs

def print_range(tensor):
    # Find the minimum and maximum values in the tensor
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)

    # Print the range of the tensor
    print(f"Range of tensor: [{min_val}, {max_val}]")

def get_total_loss(args, data, output, loss_from_g, f_y, dual_outputs, rho):
    # Feasible value 
    satisfied_loss = torch.mean(f_y(data, output), dim=1)
    if not args.not_class:
        # class 
        g_x_loss = (loss_from_g - 0.5)
    else:
        # not class
        g_x_loss = -(loss_from_g - 0.5)
    g_x_loss = torch.nn.functional.relu(g_x_loss)
    g_x_loss = torch.squeeze(g_x_loss)
    dual_outputs = torch.squeeze(dual_outputs)
    if torch.any(torch.lt(dual_outputs, 0)):
        print("dual_outputs is less than 0")
        exit()
    g_x_loss_for_satisfied = torch.mul(g_x_loss.clone(), dual_outputs)
    satisfied_loss += g_x_loss_for_satisfied
    # Multiply by rho
    penalty = torch.pow(g_x_loss, 2) * (rho/2)
    # non_satisfied_loss = satisfied_loss + penalty + g_x_loss_for_satisfied
    non_satisfied_loss = satisfied_loss + penalty
    return satisfied_loss, non_satisfied_loss

def train_primal(primal, dual, data, train_loader, n_theta, loss_g, f_y, optimizer_primal, rho, batch_idx, epoch, args, device):
    optimizer_primal.zero_grad()
    output = primal(data)
    output = torch.sigmoid(output)
    dual_outputs = dual(data)
    loss_from_g, outputs_of_g = loss_from_g_alpha(
        output, loss_g, n_theta, args, device)
    satisfied_loss, non_satisfied_loss = get_total_loss(
        args, data, output, loss_from_g, f_y, dual_outputs, rho)
    val_g_y = outputs_of_g.clone().detach()
    if not args.not_class:
        # For class (e.g. 9) case
        mask = (val_g_y >= 0.5).float()
    else:
        # For not class (e.g. 9) case
        mask = (val_g_y < 0.5).float()
    mask = torch.squeeze(mask)
    result = torch.ones_like(satisfied_loss) - 2*mask
    loss = torch.where(result == 1, satisfied_loss, non_satisfied_loss,)
    loss = loss.mean()
    loss.backward()
    nn.utils.clip_grad_norm_(primal.parameters(), max_norm=1.0)
    optimizer_primal.step()
    final_loss = loss
    if batch_idx % args.log_interval == 0:
        logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\t Total Loss: {:.6f}  \t Satisfied Loss: {:.6f} \t Non Satisfied Loss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), final_loss.item(), satisfied_loss.mean().item(), (non_satisfied_loss).mean().item()))
        logger.info(f"Total number of constraints: {mask.shape[0]}")
        logger.info(f"Total number of non satisfied constraints: {torch.sum(mask)}")

def train_dual(primal, dual, copy_of_dual, data, n_theta, optimizer_dual, loss_g, rho, args, batch_idx, device):
    l2_loss = nn.MSELoss()
    optimizer_dual.zero_grad()
    primal_output = primal(data)
    primal_output = torch.sigmoid(primal_output)
    loss_from_g, outputs_of_g = loss_from_g_alpha(primal_output, loss_g, n_theta, args, device)
    dual_outputs_frozen = copy_of_dual(data)
    current_iter_dual_outputs = dual(data)
    if not args.not_class:
        # class 
        non_satisfied_output = (loss_from_g - 0.5)
    else:
        # not class
        non_satisfied_output = -(loss_from_g - 0.5)

    first_tensor = dual_outputs_frozen + rho * non_satisfied_output
    second_tensor = torch.zeros_like(first_tensor)
    second_tensor[first_tensor > 0] = first_tensor[first_tensor > 0]
    loss = l2_loss(current_iter_dual_outputs, second_tensor)
    loss = loss.mean()
    loss.backward()
    nn.utils.clip_grad_norm_(dual.parameters(), max_norm=1.0)
    optimizer_dual.step()
    if batch_idx % args.log_interval == 0:
            logger.info('Dual Train :\t Loss: {:.6f} '.format(loss.item()))
            

def train(args, primal, dual, n_theta, device, train_loader, optimizer_primal, optimizer_dual, f_y, loss_g, epoch):
    primal.train()
    dual.train()
    n_theta.eval()
    rho = args.rho
    for batch_idx, (data, target) in enumerate(train_loader):
        # The method is PDL
        data, _ = data.to(device), target.to(device),
        data = data.reshape(-1, 28 * 28).to(device)
        # Train Primal Model
        train_primal(primal, dual, data, train_loader, n_theta, loss_g, f_y, optimizer_primal, rho, batch_idx, epoch, args, device)
    
    copy_of_dual = copy.deepcopy(dual)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, _ = data.to(device), target.to(device),
        data = data.reshape(-1, 28 * 28).to(device)
        # Train Dual
        train_dual(primal, dual, copy_of_dual, data, n_theta, optimizer_dual, loss_g, rho, args, batch_idx, device)
        

def validate(args, model, n_theta, f_xy, device, test_loader, best_loss, counter):
    model.eval()
    n_theta.eval()
    test_loss = 0
    correct = 0
    correct_2 = 0
    # loss = BCEWithLogitsLoss()
    min_delta = 0.0005  # Minimum change in the validation loss to be considered as improvement
    loss_function = nn.MSELoss()
    with torch.no_grad():
        for data, target in test_loader:
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
            # Use f_xy to get the loss
            output = n_theta(data)
            output = torch.sigmoid(output)
            pred = (output > 0.5).float()
            correct_2 += pred.eq(target.view_as(pred)).sum().item()

    logger.info(
        '\nTest set: Average loss: {:.4f}, Adv. Accuracy: {}/{} ({:.0f}%, Actual Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct /
            len(test_loader.dataset), correct_2, len(test_loader.dataset),
            100. * correct_2 / len(test_loader.dataset)))
    if test_loss < best_loss - min_delta:
        best_loss = test_loss
        counter = 0
    else:
        counter += 1
    # correct - for examples generated adversarially
    adv_accuracy = 100. * correct / len(test_loader.dataset)
    true_accuracy = 100. * correct_2 / len(test_loader.dataset)
    # correct_2 - Input example images
    return best_loss, test_loss,  counter, adv_accuracy, true_accuracy


def generate_output_images(model, test_loader, device, output_dir):
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
        for data, target in test_loader:
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
        wandb.init(project=f"pdl_{select_class}_{class_label}")
        wandb.config.update(args)
    # Define the output directory
    if not args.not_class:
        name = f"{args.class_index}/{args.class_index}"
    else:
        name = f"{args.class_index}/not_{args.class_index}"
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

    logger.info(f"You have selected {select_class}")
    logger.info(f"You have selected {class_label}")

    if args.dataset.lower() == 'mnist':
        loaders = get_MNIST_data_loaders_test(args, n_theta, n_theta, train_kwargs, test_kwargs,
                                                                                     select_class=select_class, class_label=class_label, device=device)
    if not args.not_class:
        loader_for_1 = loaders['1']
        train_loader, test_loader = loader_for_1
    else:
        loader_for_0 = loaders['0']
        train_loader, test_loader= loader_for_0


    input_batch, _ = next(iter(train_loader))
    # extract the number of features
    logger.info(f"{input_batch.shape[2]}, {input_batch.shape[3]}")
    input_size = input_batch.shape[2] * input_batch.shape[3]
    hidden_size = 512
    # Load the dataset
    # Initialize the autoencoder model for adversarial training
    primal = Autoencoder(input_size,).to(device)
    dual = DualModel(input_size,).to(device)
    # Define the loss function and optimizer
    optimizer_primal = optim.Adam(primal.parameters(), lr=args.lr)
    optimizer_dual = optim.Adam(dual.parameters(), lr=args.lr)

    # Define the learning rate scheduler
    lr_scheduler_primal = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_primal, mode='min', factor=args.gamma, threshold=1e-3, patience=2)
    lr_scheduler_dual = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_dual, mode='min', factor=args.gamma, threshold=1e-3, patience=2)
    # loss_ae = nn.MSELoss(reduction='none')
    # Define Loss from a PGM 
    
    # The values represent weights for (0,0), (0,1), (1,0), (1,1)
    negative_weight = 1e-2
    positive_weight = 2.0
    factor_f = torch.tensor([negative_weight, positive_weight , positive_weight, negative_weight]).to(device)
    f_xy = nn.DataParallel(f_xy_FunctionEvaluationModel(factor_f))
    loss_ae = f_xy
    
    # loss_ae = nn.L1Loss()
    loss_g = LossFromK()
    best_loss = float('inf')
    counter = 0
    patience = 15  # Number of epochs to wait for the validation loss to improve
    # if not debug:
    #     _, _, adv_accuracy, true_accuracy  = validate(args, autoencoder, n_theta, device, test_loader, best_loss, counter)
    #     _, _, adv_accuracy, true_accuracy  = validate(args, autoencoder, n_theta, device, train_loader, best_loss, counter)

    for epoch in range(1, args.ae_epochs + 1):
        batch_size = train_loader.batch_size
        train(args, primal, dual, n_theta, device,
              train_loader, optimizer_primal, optimizer_dual, loss_ae, loss_g, epoch)
        # _, _, _, adv_accuracy, true_accuracy = validate(
        #     args, primal, n_theta, device, train_loader, best_loss, counter)
        best_loss, test_loss, counter, adv_accuracy, true_accuracy = validate(
            args, primal, n_theta, f_xy, device, test_loader, best_loss, counter)
        lr_scheduler_primal.step(test_loss)
        lr_scheduler_dual.step(test_loss)
        print("Learning rate:", optimizer_primal.param_groups[0]['lr'])
        
        # if counter >= patience:
        #     print("Validation loss hasn't improved for {} epochs, stopping training...".format(patience))
        #     break
    # Save Actual and Adversarial Images
    if not args.not_class:
        name = f"{args.class_index}/{args.class_index}"
    else:
        name = f"{args.class_index}/not_{args.class_index}"
    if args.binarize:
        all_inputs, all_outputs = generate_output_images(primal, test_loader, device, output_dir.replace("models", "adv_images_binarized"))
    else:
        all_inputs, all_outputs = generate_output_images(primal, test_loader, device, output_dir.replace("models", "adv_images_continous"))
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
        torch.save(lr_scheduler_primal.state_dict(),
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
    parser.add_argument('--binarize', action='store_true', default=False,
                        help='Binarize the dataset?')
    parser.add_argument('--use_nn', action='store_true', default=False,
                        help='Use the larger model?')
    parser.add_argument(
        '--dataset', choices=['MNIST', 'EMNIST'], default='MNIST', help='Choose a dataset')
    parser.add_argument('--model', type=str, metavar='model',
                        help='Location of model')
    parser.add_argument('--class-index', type=int, metavar='C',
                        help='provide the class to train the model')
    parser.add_argument('--not-class', action='store_true', default=False,
                        help='Which class of the sigmoid to test for - if given test for the not class (not 9) else for class (9)')
    parser.add_argument('--rho', type=float, default=20, metavar='LR',
                        help='learning rate (default: 1.0)')
    args = parser.parse_args()
    pprint(args)
    train_ae(args, None)
