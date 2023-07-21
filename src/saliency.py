import gzip
import os
import shutil
import sys

import nibabel as nib
import numpy as np
import torch.utils.data
import torchio as tio
from captum.attr import Saliency

from dcan.data_sets.dsets import LoesScoreDataset
from reprex.models import AlexNet3D


def compute_saliency(nifti_input):
    def normalize_array(array):
        new_array = (array - array.min()) / (array.max() - array.min())

        return new_array

    net = AlexNet3D(4608)
    net.eval()

    image = tio.ScalarImage(nifti_input)

    image_tensor = image.data

    image_tensor = torch.unsqueeze(image_tensor, dim=0)
    image_tensor = normalize_array(image_tensor)

    with torch.no_grad():
        output = net(image_tensor)
        print(output)

    print("Using existing trained model")
    net.load_state_dict(torch.load('models/loes_scoring_03.pt',
                                   map_location='cpu'))

    csv_data_file = "./data/MNI-space_Loes_data.csv"
    testset = LoesScoreDataset(
        csv_data_file,
        use_gd_only=False,
        val_stride=10,
        is_val_set_bool=True,
    )
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    dataiter = iter(testloader)
    images, labels = next(dataiter)

    ind = 3

    img_input = images[ind].unsqueeze(0)
    img_input.requires_grad = True

    net.eval()

    def wrapped_model(inp):
        return net(inp)[0]

    saliency = Saliency(wrapped_model)
    grads = saliency.attribute(img_input)
    grads = grads.squeeze().cpu().detach().numpy()

    return grads


if __name__ == '__main__':
    my_nifti_input = sys.argv[1]
    img = nib.load(my_nifti_input)
    saliency_output_file_path = sys.argv[2]
    shutil.copyfile(my_nifti_input, saliency_output_file_path)
    array_data = compute_saliency(my_nifti_input)
    normalized_vector = array_data / np.linalg.norm(array_data)
    scaled_vector = np.dot(normalized_vector, 256)
    affine = img.affine
    array_img = nib.Nifti1Image(scaled_vector, affine)
    nii_out_file = saliency_output_file_path[:-3]
    nib.save(array_img, nii_out_file)
    with open(nii_out_file, 'rb') as f_in:
        with gzip.open(saliency_output_file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(nii_out_file)
