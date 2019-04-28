import torch
import torch.nn as nn
from torch.utils.data import Dataset

class sphere_dataset(Dataset):

    def __init__(self, R1=1, R2=1.3, dim=500, n_elem=50000000):
        """ initializes parameters """
        super().__init__()
        self.radii = [R1,R2]
        self.dim = dim
        self.n_elem = n_elem

    def __len__(self):
        return self.n_elem

    def __getitem__(self, idx):
        # we will alternate class types
        class_type = idx % 2

        x = torch.randn(self.dim)
        return (x/torch.norm(x, p=2))*self.radii[class_type], class_type

class single_sphere_dataset(Dataset):
    """ samples points uniformly from a single sphere
    of given dimension and radius """

    def __init__(self, radius=1, dim=500, n_elem=50000000):
        """ initializes parameters """
        super().__init__()
        self.radius = radius 
        self.dim = dim
        self.n_elem = n_elem

    def __len__(self):
        return self.n_elem

    def __getitem__(self, idx):
        # we will alternate class types

        x = torch.randn(self.dim)
        return (x/torch.norm(x, p=2))*self.radius
