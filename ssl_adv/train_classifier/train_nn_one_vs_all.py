from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Module
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss

import wandb
from loguru import logger
from torch.utils.data import default_collate
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from utils import create_directory, get_date_as_string


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


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    # loss = CrossEntropyLoss()
    loss = BCEWithLogitsLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.reshape(-1, 28 * 28).to(device)
        optimizer.zero_grad()
        output = model(data)
        target = target.unsqueeze(1)
        loss_value = loss(output, target.float())
        loss_value.backward()
        optimizer.step()
        wandb.log({f"train_loss": loss_value.item()}, )
        if batch_idx % args.log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss_value.item()))
            if args.dry_run:
                break


def validate(model, device, test_loader, best_loss, counter):
    model.eval()
    test_loss = 0
    correct = 0
    loss = BCEWithLogitsLoss()
    min_delta = 0.01  # Minimum change in the validation loss to be considered as improvement

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.reshape(-1, 28 * 28).to(device)
            output = model(data)
            target = target.unsqueeze(1)
            test_loss += loss(output, target.float()).item()  # sum up batch loss
            output = F.sigmoid(output)
            pred = (output > 0.5).float()
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    wandb.log({f"test_loss": test_loss}, )
    wandb.log({f"test_accuracy": 100. * correct / len(test_loader.dataset)}, )

    logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    if test_loss < best_loss - min_delta:
        best_loss = test_loss
        counter = 0
    else:
        counter += 1
    return best_loss, counter


def get_MNIST_data_loaders(args, train_kwargs, test_kwargs, select_class):
    if not args.continous:
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
    idx = train_data.targets != select_class
    print(f"Number of examples in the dataset: {len(train_data.targets)}")
    print(f"Number of examples in the dataset for other classes: {len(train_data.targets[idx])}")
    print(f"Number of examples in the dataset for class {select_class}: {len(train_data.targets[~idx])}")
    idx = train_data.targets != select_class
    train_data.targets[idx] = 0
    idx2 = train_data.targets != 0
    train_data.targets[idx2] = 1
    train_loader = torch.utils.data.DataLoader(train_data, **train_kwargs
                                               )
    test_data = datasets.MNIST('../../data', train=False, transform=transform)
    idx = test_data.targets != select_class
    idx = test_data.targets != select_class
    test_data.targets[idx] = 0
    idx2 = test_data.targets != 0
    test_data.targets[idx2] = 1
    print(f"Number of examples in the dataset: {len(test_data.targets)}")
    print(f"Number of examples in the dataset for other classes: {len(test_data.targets[idx])}")
    print(f"Number of examples in the dataset for class {select_class}: {len(test_data.targets[~idx])}")
    test_loader = torch.utils.data.DataLoader(test_data, **test_kwargs)
    return train_loader, test_loader



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1024, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
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
    parser.add_argument('--use-lr', action='store_true', default=False,
                        help='Use the smaller model?')
    parser.add_argument('--binarize', action='store_true', default=False,
                        help='Binarize the dataset?')
    parser.add_argument('--class-index', type=int, default=0, metavar='C',
                        help='provide the class to train the model')
    parser.add_argument('--model-location', type=str, default=None,  metavar='loc',
                        help='Location where the models needs to be saved')
    parser.add_argument('--dataset', choices=['MNIST', 'EMNIST'], default='MNIST', help='Choose a dataset')
    args = parser.parse_args()
    args.continous = not args.binarize
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()
    
    import sys
    wandb.init(project=f"Train Classifier {args.dataset} {args.class_index}")
    wandb.config.update(args)
    # Add a logger to the project
    config = {
        "handlers": [
            {"sink": sys.stdout,
             "format": "<green>{time:YYYY-MM-DD at HH:mm:ss}</green> | {module}.{function} | <level>{message}</level> "},
            {"sink": "logging/logger_{time}.log",
             "format": "<green>{time:YYYY-MM-DD at HH:mm:ss}</green> | {module}.{function} | <level>{message}</level> "}
        ],
        "extra": {"user": "sva"}
    }
    logger.configure(**config)
    # torch.manual_seed(args.seed)
    if use_cuda:
        logger.info("Using GPU for training")

    input_size = 28 * 28
    num_classes = 1
    num_epochs = args.epochs
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
    selected_class = int(args.class_index)
    if args.dataset.lower() == 'mnist':
        train_loader, test_loader = get_MNIST_data_loaders(args, train_kwargs, test_kwargs, select_class=selected_class)
    else:
        raise ValueError("Dataset not supported")
    layer1, layer2 = 128, 256
    if args.use_lr:
        logger.info("Using the logistic regression model")
        model = LR(input_size, num_classes).to(device)
    else:
        logger.info("Using the NN model")
        model = Net(input_size, [(layer1, nn.ReLU()), (layer2, nn.ReLU()), (num_classes, None)], )
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = StepLR(optimizer, step_size=num_epochs // 5, gamma=args.gamma)
    best_loss = float('inf')
    counter = 0
    patience = 15  # Number of epochs to wait for the validation loss to improve
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        _, _ = validate(model, device, train_loader, 0, 0)
        best_loss, counter = validate(model, device, test_loader, best_loss, counter)
        scheduler.step()
        # if counter >= patience:
        #     print("Validation loss hasn't improved for {} epochs, stopping training...".format(patience))
        #     break
    if args.save_model:
        date = get_date_as_string()
        date = date + f"selected_class_{selected_class}"
        save_location = ""
        if args.use_lr:
            if not args.continous:
                create_directory("../trained_classifier/lr_binarized")
                save_location = f"../trained_classifier/lr_binarized/{args.dataset.lower()}_{args.class_index}_nn.pt"
                torch.save(model, save_location)
            else:
                create_directory("../trained_classifier/lr")
                save_location = f"../trained_classifier/lr/{args.dataset.lower()}_{args.class_index}_nn.pt"
                torch.save(model, save_location)
        else:
            if not args.continous:
                create_directory("../trained_classifier/nn_binarized")
                save_location = f"../trained_classifier/nn_binarized/{args.dataset.lower()}_{args.class_index}_nn.pt"
                torch.save(model, save_location)
            else:
                create_directory("../trained_classifier/nn")
                save_location = f"../trained_classifier/nn/{args.dataset.lower()}_{args.class_index}_nn.pt"
                torch.save(model, save_location)

        print(save_location)
if __name__ == '__main__':
    main()
