import torch.nn as nn

# Define the autoencoder model
# Define the neural network architecture

class LR(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LR, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        return out

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(NeuralNetwork, self).__init__()
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_sizes)):
            if i == 0:
                self.hidden_layers.append(nn.Linear(input_size, hidden_sizes[i]))
            else:
                self.hidden_layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            self.hidden_layers.append(nn.ReLU())
        self.output = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        model_output = self.output(x)
        return model_output
