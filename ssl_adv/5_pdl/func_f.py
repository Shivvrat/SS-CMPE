from torch import nn
import torch


class f_xy_FunctionEvaluationModel(nn.Module):
    def __init__(self, func_xy):
        super().__init__()
        self.func_xy = nn.Parameter(func_xy)
        self.func_xy.requires_grad = False

    def forward(self, x, y):
        f_y = 0
        
        # y[y < 0.5] = 0
        # y[y >= 0.5] = 1
        
        f_y += self.func_xy[0]*(1 - x)*(1 - y)
        f_y += self.func_xy[1]*(1 - x)*(y)
        f_y += self.func_xy[2]*(x)*(1 - y)
        f_y += self.func_xy[3]*(x)*(y)
        return f_y