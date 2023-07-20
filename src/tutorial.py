import numpy as np

import torch
import torch.nn as nn

from captum.attr import IntegratedGradients

# Create and prepare model:
class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(3, 3)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(3, 2)

        # initialize weights and biases
        self.lin1.weight = nn.Parameter(torch.arange(-4.0, 5.0).view(3, 3))
        self.lin1.bias = nn.Parameter(torch.zeros(1,3))
        self.lin2.weight = nn.Parameter(torch.arange(-3.0, 3.0).view(2, 3))
        self.lin2.bias = nn.Parameter(torch.ones(1,2))

    def forward(self, input):
        return self.lin2(self.relu(self.lin1(input)))


model = ToyModel()
model.eval()

# To make computations deterministic, let's fix random seeds:
torch.manual_seed(123)
np.random.seed(123)

# Define input and baseline tensors:
input = torch.rand(2, 3)
baseline = torch.zeros(2, 3)

# Select algorithm to instantiate and apply (Integrated Gradients in this example):
ig = IntegratedGradients(model)
attributions, delta = ig.attribute(input, baseline, target=0, return_convergence_delta=True)
print('IG Attributions:', attributions)
print('Convergence Delta:', delta)