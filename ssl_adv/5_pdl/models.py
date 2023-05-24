import torch.nn as nn


# Define the autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, input_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
                nn.Linear(input_size, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
        )
        self.decoder = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, input_size),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class DualModel(nn.Module):
    def __init__(self, input_size, output_size=1):
        super(DualModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        # self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(128, output_size)
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
                nn.init.normal_(m.weight)
    
    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        # x = nn.functional.relu(self.fc2(x))
        out = self.fc3(x)
        out = nn.functional.relu(out)
        return out
