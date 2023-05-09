import torch
import torch.nn.functional as F
import torchio as tio

from PIL import Image

import os
import json
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz

from reprex.models import AlexNet3D
import nibabel as nib


def normalize_array(array):
    new_array = (array - array.min()) / (array.max() - array.min())

    return new_array

model = AlexNet3D(4608)
model_save_location = '/home/miran045/reine097/projects/MyCaptum/models/loes_scoring_03.pt'
model.load_state_dict(torch.load(model_save_location,
                                     map_location='cpu'))
model.eval()

example_file = '/home/feczk001/shared/projects/S1067_Loes/data/MNI-space_Loes_data/sub-4750MASZ_ses-20080220_space-MNI_mprageGd.nii.gz'
image = tio.ScalarImage(example_file)

image_tensor = image.data

image_tensor = torch.unsqueeze(image_tensor, dim=0)
image_tensor = normalize_array(image_tensor)

with torch.no_grad():
    output = model(image_tensor)
    print(output)
