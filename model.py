import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO Implement batch norm

class SmallReLU(nn.Module):
    
    def __init__(self, sphere_dim, n_hidden):
        super().__init__()

        self.L1 = nn.Linear(in_features=sphere_dim,
                            out_features=n_hidden)
        self.L2 = nn.Linear(in_features=n_hidden,
                            out_features=n_hidden)
        self.L3 = nn.Linear(in_features=n_hidden,
                            out_features=1, bias=False)

    def forward(self, x):
        x = F.relu(self.L1(x))
        x = F.relu(self.L2(x))
        x = self.L3(x)

        return x

class LargeReLU(nn.Module):
    
    def __init__(self, sphere_dim, n_hidden):
        super().__init__()

        self.L1 = nn.Linear(in_features=sphere_dim,
                            out_features=n_hidden)
        self.L2 = nn.Linear(in_features=n_hidden,
                            out_features=n_hidden)
        self.L3 = nn.Linear(in_features=n_hidden,
                            out_features=n_hidden)
        self.L4 = nn.Linear(in_features=n_hidden,
                            out_features=n_hidden)
        self.L5 = nn.Linear(in_features=n_hidden,
                            out_features=n_hidden)
        self.L6 = nn.Linear(in_features=n_hidden,
                            out_features=1, bias=False)

    def forward(self, x):
        x = F.relu(self.L1(x))
        x = F.relu(self.L2(x))
        x = F.relu(self.L3(x))
        x = F.relu(self.L4(x))
        x = F.relu(self.L5(x))
        x = self.L6(x)

        return x


