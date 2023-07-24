import sys
import pandas as pd
from tqdm import tqdm
import os
import numpy as np
import nibabel as nib
import gzip
import shutil

from dcan.explainability.saliency import compute_saliency


def compute_saliency_from_spreadsheet(spreadsheet_file_path, model_file_path, output_folder):
    df = pd.read_csv(spreadsheet_file_path)
    df = df.reset_index()
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for _, row in tqdm(df.iterrows()):
        input_file = row['file']
        file_name = os.path.split(input_file)[1]
        base_file_name = file_name[:-7]
        saliency_output_file_name = os.path.join(output_folder, f'{base_file_name}_salience.nii.gz')
        if os.path.exists(saliency_output_file_name):
            continue
        saliency_grads = compute_saliency(input_file, model_file_path)
        normalized_vector = saliency_grads / np.linalg.norm(saliency_grads)
        scaled_vector = np.dot(normalized_vector, 256)
        img = nib.load(input_file)
        affine = img.affine
        array_img = nib.Nifti1Image(scaled_vector, affine)
        nii_out_file = saliency_output_file_name[:-3]
        nib.save(array_img, nii_out_file)
        with open(nii_out_file, 'rb') as f_in:
            with gzip.open(saliency_output_file_name, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(nii_out_file)


if __name__ == '__main__':
    spreadsheet_file_path = sys.argv[1]
    model_file_path = sys.argv[2]
    output_folder = sys.argv[3]
    compute_saliency_from_spreadsheet(spreadsheet_file_path, model_file_path, output_folder)
