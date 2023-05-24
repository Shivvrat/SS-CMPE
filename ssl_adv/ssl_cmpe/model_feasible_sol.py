# Define your neural network architecture
import torch.nn as nn
import torch


class FeasibleNet(nn.Module):
    def __init__(self, input_size, hidden_size, class_1, device):
        super(FeasibleNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, input_size)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.device = device
        self.optimize_class_1 = class_1

        
    def forward(self, x):
        x = x.reshape(-1, 28 * 28).to(self.device)
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        x = torch.sigmoid(x)
        return x
    
    def train(self, data, func_g, epochs=50):
        # Train your neural network
        for epoch in range(epochs):
            # for data, target in data_loader:
                running_loss = 0.0
                self.optimizer.zero_grad()
                # Forward pass
                y_pred = self.forward(data.to(self.device))
                g_y = func_g(y_pred).mean()
                if not self.optimize_class_1:
                    print("We are optimizing for class 0")
                    g_y = -g_y
                # Backward pass and optimization
                g_y.backward()
                self.optimizer.step()
                running_loss += g_y.item()

    def generate_feasible_examples(self, X_train):
        with torch.no_grad():
            outputs = self.forward(X_train)
            assert (outputs > 0).any(),"model not trained"
        return outputs
