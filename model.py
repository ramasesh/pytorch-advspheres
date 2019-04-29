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

class QuadraticNetwork(nn.Module):
    """ class for the network with a quadratic point-wise linearity """

    def __init__(self, sphere_dim=500, n_hidden=1000):
        super().__init__()
        self.Linear1 = nn.Linear(in_features=sphere_dim,
                                 out_features=n_hidden,
                                 bias=False)
        self.Linear2 = nn.Linear(in_features=1, 
                                 out_features=1,
                                 bias=True)

        self.update_alphas()

    def forward(self, x):
        x = self.Linear1(x)
        x = x * x
        x = torch.sum(x, dim=-1).view(-1,1)
        x = self.Linear2(x)

        return x

    def update_alphas(self):
        """ the alpha parameters given in the adv spheres paper
        are the eigenvalues of W^T W, where W^T is the weight matrix
        of the first linear layer """
        
        W = self.Linear1.weight.data
        w2 = self.Linear2.weight.data
        b2 = self.Linear2.bias.data
        with torch.no_grad():
            WTW = torch.matmul(W.t(), W)
            self.alphas = (-1*torch.symeig(WTW)[0]*w2/b2)[0] 
