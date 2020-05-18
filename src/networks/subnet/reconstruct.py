import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import sys
import numpy as np

from networks.utils import conv3x3

class ReconstructNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ReconstructNet, self).__init__()
        self.reconstruct = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(True),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(True),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        # x : the weight matrix that has been applied dropout with the eigen
        return self.reconstruct(x)
