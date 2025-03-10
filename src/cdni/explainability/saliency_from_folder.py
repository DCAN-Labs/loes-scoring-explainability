import sys
import os

from tqdm import tqdm
# Add the parent directory of dcan to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Now try importing
from cdni.explainability.saliency_visualization import create_saliency_nifti

def create_saliency_maps(nifti_input_folder, saliency_output_folder, model_file_path):
    files = os.listdir(nifti_input_folder)
    for file in tqdm(files):
        saliency_output_file_path = os.path.join(saliency_output_folder, file)
        saliency_input_file_path = os.path.join(nifti_input_folder, file)
        create_saliency_nifti(saliency_input_file_path, saliency_output_file_path, model_file_path)


if __name__ == '__main__':
    nifti_input_folder = sys.argv[1]
    saliency_output_folder = sys.argv[2]
    model_file_path = sys.argv[3]

    create_saliency_maps(nifti_input_folder, saliency_output_folder, model_file_path)
