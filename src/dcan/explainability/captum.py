import torch
import torch.nn.functional as F

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

from reprex.models import AlexNet3DDropoutRegression
import nibabel as nib

model = AlexNet3DDropoutRegression(9600)
model_save_location = '/home/miran045/reine097/projects/MyCaptum/models/loes_scoring_02.pt'
model.load_state_dict(torch.load(model_save_location,
                                 map_location='cpu'))
model.eval()

example_file = '/home/feczk001/shared/projects/S1067_Loes/data/MNI-space_Loes_data/sub-4750MASZ_ses-20080220_space-MNI_mprageGd.nii.gz'
img = nib.load(example_file)
image_data = img.get_fdata()

output = model(image_data)
