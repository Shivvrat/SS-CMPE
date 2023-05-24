import torch

import torch
import torch.nn as nn

class PGMLoss(nn.Module):
    def __init__(self, univariate_functions, bivariate_functions, device):
        super(PGMLoss, self).__init__()
        self.univariate_weights_0 = torch.tensor([func[1] for func in univariate_functions], requires_grad=False).to(device)
        self.univariate_weights_1 = torch.tensor([func[2] for func in univariate_functions], requires_grad=False).to(device)
        self.univariate_vars = torch.tensor([func[0] for func in univariate_functions], dtype=torch.long, requires_grad=False).to(device)

        self.bivariate_weights_00 = torch.tensor([func[2] for func in bivariate_functions], requires_grad=False).to(device)
        self.bivariate_weights_01 = torch.tensor([func[3] for func in bivariate_functions], requires_grad=False).to(device)
        self.bivariate_weights_10 = torch.tensor([func[4] for func in bivariate_functions], requires_grad=False).to(device)
        self.bivariate_weights_11 = torch.tensor([func[5] for func in bivariate_functions], requires_grad=False).to(device)
        self.bivariate_vars_1 = torch.tensor([func[0] for func in bivariate_functions], dtype=torch.long, requires_grad=False).to(device)
        self.bivariate_vars_2 = torch.tensor([func[1] for func in bivariate_functions], dtype=torch.long, requires_grad=False).to(device)

    def forward(self, X, y):
        total_data_tensor = torch.cat((X, y), dim=1)
        univariate_contributions = (1 - total_data_tensor[:, self.univariate_vars]) * self.univariate_weights_0 + \
                                   total_data_tensor[:, self.univariate_vars] * self.univariate_weights_1
        bivariate_contributions = (1 - total_data_tensor[:, self.bivariate_vars_1]) * (1 - total_data_tensor[:, self.bivariate_vars_2]) * self.bivariate_weights_00.unsqueeze(0) + \
                                  (1 - total_data_tensor[:, self.bivariate_vars_1]) * total_data_tensor[:, self.bivariate_vars_2] * self.bivariate_weights_01.unsqueeze(0) + \
                                  total_data_tensor[:, self.bivariate_vars_1] * (1 - total_data_tensor[:, self.bivariate_vars_2]) * self.bivariate_weights_10.unsqueeze(0) + \
                                  total_data_tensor[:, self.bivariate_vars_1] * total_data_tensor[:, self.bivariate_vars_2] * self.bivariate_weights_11.unsqueeze(0)
        loss_val = torch.sum(univariate_contributions, dim=1) + torch.sum(bivariate_contributions, dim=1)
        return loss_val
