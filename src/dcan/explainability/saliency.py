# Authors: Anders Perrone and Paul Reiners

import gzip
import os
import shutil
import sys

import nibabel as nib
import numpy as np
import torch.utils.data
import torchio as tio
from captum.attr import Saliency

from reprex.models import AlexNet3D


def normalize_array(array):
    new_array = \
        torch.subtract(array, torch.min(array)) / \
        torch.subtract(torch.max(array), torch.min(array))

    return new_array


def compute_saliency(nifti_input, model):
    net = AlexNet3D(4608)
    net.eval()

    img = tio.ScalarImage(nifti_input)

    image_tensor = img.data

    image_tensor = torch.unsqueeze(image_tensor, dim=0)
    image_tensor = normalize_array(image_tensor)
    image_tensor.requires_grad = True

    net.load_state_dict(torch.load(model,
                                   map_location='cpu'))

    net.eval()

    def wrapped_model(inp):
        return net(inp)[0]

    saliency = Saliency(wrapped_model)
    grads = saliency.attribute(image_tensor)
    grads = grads.squeeze().cpu().detach().numpy()

    return grads


def create_nifti(img, scaled_vector, saliency_output_file_name):
    affine = img.affine
    array_img = nib.Nifti1Image(scaled_vector, affine)
    nii_out_file = saliency_output_file_name[:-3]
    nib.save(array_img, nii_out_file)
    with open(nii_out_file, 'rb') as f_in:
        with gzip.open(saliency_output_file_name, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    return nii_out_file


if __name__ == '__main__':
    my_nifti_input = sys.argv[1]
    image = nib.load(my_nifti_input)
    saliency_output_file_path = sys.argv[2]
    model_file_path = sys.argv[3]
    shutil.copyfile(my_nifti_input, saliency_output_file_path)
    array_data = compute_saliency(my_nifti_input, model_file_path)
    normalized_vector = array_data / np.linalg.norm(array_data)
    nii_out_fl = create_nifti(image, np.dot(normalized_vector, 256), saliency_output_file_path)
    os.remove(nii_out_fl)
