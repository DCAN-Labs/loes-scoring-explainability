import os
import sys

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torchio as tio
from tqdm import tqdm

from dcan.explainability.saliency import normalize_array
from reprex.models import AlexNet3D


def compute_saliency_from_spreadsheet(
        spreadsheet_file_path, model_file_path, output_folder, output_spreadsheet_file_name):
    df, output_df = create_dfs(spreadsheet_file_path)
    model = AlexNet3D(4608)
    model.eval()
    model.load_state_dict(torch.load(model_file_path, map_location='cpu'))

    for index, row in tqdm(df.iterrows()):
        input_file = row['file']

        image = tio.ScalarImage(input_file)
        image_tensor = image.data
        image_tensor = torch.unsqueeze(image_tensor, dim=0)
        image_tensor = normalize_array(image_tensor)
        image_tensor.requires_grad = True
        output = model(image_tensor)
        prediction = output[0].item()
        output_df.at[index, 'prediction'] = prediction
        output_df.at[index, 'error'] = abs(prediction - output_df.at[index, 'loes-score'])

        saliency_output_file_name = get_saliency_file_name(index, input_file, output_df, output_folder)
        output_df.at[index, 'saliency-map-file'] = saliency_output_file_name
    output_df = output_df.sort_values(by='error')
    output_df.to_csv(output_spreadsheet_file_name)


def get_saliency_file_name(index, input_file, output_df, output_folder):
    file_name = os.path.split(input_file)[1]
    base_file_name = file_name[:-7]
    saliency_output_file_name = os.path.join(output_folder, f'{base_file_name}_salience.nii.gz')
    img = nib.load(saliency_output_file_name)
    saliency_grads = img.get_fdata()
    update_df_with_saliency_columns(index, output_df, saliency_grads)
    return saliency_output_file_name


def update_df_with_saliency_columns(index, output_df, saliency_grads):
    normalized_vector = saliency_grads / np.linalg.norm(saliency_grads)
    scaled_vector = np.dot(normalized_vector, 256)
    # noinspection PyArgumentList
    output_df.at[index, 'max-saliency'] = scaled_vector.max()
    ind = np.unravel_index(np.argmax(scaled_vector, axis=None), scaled_vector.shape)
    output_df.at[index, 'max-saliency-x'] = int(ind[0])
    output_df.at[index, 'max-saliency-y'] = int(ind[1])
    output_df.at[index, 'max-saliency-z'] = int(ind[2])
    output_df.at[index, 'saliency-map-file'] = ''
    return scaled_vector


def create_dfs(spreadsheet_file_path):
    df = pd.read_csv(spreadsheet_file_path)
    df = df.reset_index()
    output_df = df.copy()
    output_df['max-saliency'] = np.nan
    output_df['prediction'] = np.nan
    output_df['error'] = np.nan
    output_df['max-saliency-x'] = -1
    output_df['max-saliency-y'] = -1
    output_df['max-saliency-z'] = -1
    output_df = output_df.astype({"max-saliency-x": "int", "max-saliency-y": "int", "max-saliency-z": "int"})
    return df, output_df


if __name__ == '__main__':
    compute_saliency_from_spreadsheet(
        sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
