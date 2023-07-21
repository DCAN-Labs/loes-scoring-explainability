import sys
import time

import torch.utils.data
import torchio as tio
from captum.attr import Saliency
import matplotlib.pyplot as plt
import numpy as np
import os
import nibabel as nib

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
    result = compute_saliency(my_nifti_input)
    print(result)
    print(type(result))
    print(result.max())
    size = len(result)
    img = nib.load(my_nifti_input)
    image_data = img.get_fdata()
    fig = plt.figure()
    for j in range(3):
        for i in range(0, size, 5):
            print(f'axis: {j}; slice: {i}')
            cross_section = result[i]

            plt.imshow(cross_section, cmap='hot', interpolation='nearest')
            plt.imshow(image_data[i], 'gray', interpolation='none', alpha=0.5)
            plt.show()

            time.sleep(2)
            plt.clf()

        result = np.transpose(result, (1, 2, 0))
        image_data = np.transpose(image_data, (1, 2, 0))
        size = len(result)
