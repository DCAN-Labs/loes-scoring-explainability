import numpy as np

import torch
import torchio as tio
import tensorflow as tf

from captum.attr import IntegratedGradients

# Create and prepare model
from reprex.models import AlexNet3D

def normalize_array(array):
    new_array = (array - torch.max(array)) / (torch.max(array) - torch.min(array))

    return new_array


model = AlexNet3D(4608)
model_save_location = '/home/miran045/reine097/projects/loes-scoring-explainability/models/loes_scoring_03.pt'
model.load_state_dict(torch.load(model_save_location,
                               map_location='cpu'))
model.eval()

# To make computations deterministic, let's fix random seeds:
torch.manual_seed(123)
np.random.seed(123)

# Define input and baseline tensors:
example_file = '/home/feczk001/shared/projects/S1067_Loes/data/MNI-space_Loes_data/sub-4750MASZ_ses-20080220_space-MNI_mprageGd.nii.gz'
image = tio.ScalarImage(example_file)
image_tensor = image.data
image_tensor = torch.unsqueeze(image_tensor, dim=0)
input = normalize_array(image_tensor)

baseline = torch.zeros(*list(image_tensor.shape))

# Select algorithm to instantiate and apply (Integrated Gradients in this example):
ig = IntegratedGradients(model)
attributions, delta = ig.attribute(input, baseline, target=0, return_convergence_delta=True)
print('IG Attributions:', attributions)
print('Convergence Delta:', delta)
