def validate_args(input_path, output_path, model_path):
    """
    Validate command-line arguments for the script.
    
    This function performs comprehensive validation of all input arguments
    to ensure they meet the requirements for processing.
    
    Parameters:
        input_path (str): Path to the input NIfTI file.
        output_path (str): Path where the saliency output will be saved.
        model_path (str): Path to the pre-trained model weights file.
        
    Returns:
        tuple: (is_valid, error_message) where is_valid is a boolean and
               error_message is None if valid, otherwise contains error details.
    """
    # Validate input file
    if not input_path:
        return False, "Input file path cannot be empty"
    
    if not os.path.exists(input_path):
        return False, f"Input NIfTI file not found: {input_path}"
    
    if not os.path.isfile(input_path):
        return False, f"Input path is not a file: {input_path}"
        
    # Check if input file is readable
    if not os.access(input_path, os.R_OK):
        return False, f"Input file is not readable: {input_path}"
        
    # Validate input file extension for NIfTI format
    if not (input_path.endswith('.nii') or input_path.endswith('.nii.gz')):
        return False, f"Input file does not have a NIfTI extension (.nii or .nii.gz): {input_path}"
    
    # Validate output path
    if not output_path:
        return False, "Output file path cannot be empty"
        
    # Ensure output has correct extension
    if not (output_path.endswith('.nii.gz')):
        return False, f"Output file must have .nii.gz extension: {output_path}"
    
    # Validate model file
    if not model_path:
        return False, "Model file path cannot be empty"
        
    if not os.path.exists(model_path):
        return False, f"Model file not found: {model_path}"
        
    if not os.path.isfile(model_path):
        return False, f"Model path is not a file: {model_path}"
        
    # Check if model file is readable
    if not os.access(model_path, os.R_OK):
        return False, f"Model file is not readable: {model_path}"
    
    # All checks passed
    return True, Nonedef ensure_directory_exists(directory_path):
    """
    Ensure that a directory exists and is writable.
    
    This function checks if a directory exists and creates it if necessary.
    It also verifies that the directory is writable.
    
    Parameters:
        directory_path (str): Path to the directory to check/create.
        
    Returns:
        None
        
    Raises:
        PermissionError: If the directory cannot be created or is not writable.
        IOError: If there's another issue with directory creation.
    """
    if not directory_path:
        return  # No directory to create (current directory)
        
    # Check if directory exists
    if not os.path.exists(directory_path):
        try:
            # Check if parent directory is writable before attempting to create
            parent_dir = os.path.dirname(directory_path)
            if parent_dir and not os.access(parent_dir, os.W_OK):
                raise PermissionError(f"No write permission for parent directory: {parent_dir}")
                
            # Create directory
            os.makedirs(directory_path, exist_ok=True)
        except PermissionError as pe:
            raise pe
        except Exception as e:
            raise IOError(f"Failed to create directory {directory_path}: {str(e)}")
    
    # Check write permissions on the directory
    if not os.access(directory_path, os.W_OK):
        raise PermissionError(f"No write permission for directory: {directory_path}")
# Authors: Anders Perrone and Paul Reiners
import gzip
import os
import shutil
import sys
import traceback
from dcan.models.ResNet import get_resnet_model
import nibabel as nib
import numpy as np
import torch.utils.data
import torchio as tio
from captum.attr import Saliency

def normalize_array(array):
    """
    Normalize a torch tensor to the range [0, 1] using min-max normalization.
    
    This normalization is applied to the input image before feeding it to the model,
    ensuring consistent input scale regardless of the original image intensity range.
    
    Parameters:
        array (torch.Tensor): Input tensor to be normalized.
        
    Returns:
        torch.Tensor: Normalized tensor with values between 0 and 1.
    """
    try:
        min_val = torch.min(array)
        max_val = torch.max(array)
        
        # Check for division by zero
        if max_val == min_val:
            print("Warning: Input array has constant values, returning zeros")
            return torch.zeros_like(array)
            
        new_array = \
            torch.subtract(array, min_val) / \
            torch.subtract(max_val, min_val)
        return new_array
    except Exception as e:
        print(f"Error in normalize_array: {str(e)}")
        raise

def compute_saliency(nifti_input, model):
    """
    Compute saliency map for a given NIfTI input using a pre-trained model.
    
    This function loads a ResNet model, normalizes the input image, and
    computes the saliency map using Captum's Saliency method.
    
    The input normalization (to [0,1] range) is necessary for the model to process
    the image correctly, while the output saliency map will be normalized separately
    later using L2 normalization.
    
    Parameters:
        nifti_input (str): Path to the input NIfTI file.
        model (str): Path to the pre-trained model weights file.
        
    Returns:
        numpy.ndarray: The computed saliency map as a numpy array.
    """
    try:
        # Verify input file exists
        if not os.path.exists(nifti_input):
            raise FileNotFoundError(f"Input NIfTI file not found: {nifti_input}")
            
        # Verify model file exists
        if not os.path.exists(model):
            raise FileNotFoundError(f"Model file not found: {model}")
        
        # Initialize model
        net = get_resnet_model()
        
        # Load image
        try:
            img = tio.ScalarImage(nifti_input)
        except Exception as img_error:
            raise ValueError(f"Failed to load NIfTI file: {str(img_error)}")
            
        image_tensor = img.data
        image_tensor = torch.unsqueeze(image_tensor, dim=0)
        image_tensor = normalize_array(image_tensor)
        image_tensor.requires_grad = True
        
        # Load model weights
        try:
            net.load_state_dict(torch.load(model, map_location='cpu'))
        except Exception as model_error:
            raise ValueError(f"Failed to load model weights: {str(model_error)}")
            
        # Set model to evaluation mode
        net.eval()
        
        def wrapped_model(inp):
            return net(inp)[0]
            
        saliency = Saliency(wrapped_model)
        
        # Calculate saliency
        try:
            grads = saliency.attribute(image_tensor)
            grads = grads.squeeze().cpu().detach().numpy()
            
            # Check if gradients are valid (not NaN or Inf)
            if np.isnan(grads).any() or np.isinf(grads).any():
                print("Warning: Saliency contains NaN or Inf values")
                # Replace invalid values with zeros
                grads = np.nan_to_num(grads)
                
            return grads
        except Exception as saliency_error:
            raise RuntimeError(f"Failed to compute saliency: {str(saliency_error)}")
    except Exception as e:
        print(f"Error in compute_saliency: {str(e)}")
        raise

def create_nifti(img, scaled_vector, saliency_output_file_name):
    """
    Create a NIfTI file from a numpy array using the affine matrix from a template image.
    
    This function creates a NIfTI file and saves it directly as a compressed file,
    avoiding large temporary files on disk.
    
    Parameters:
        img (nibabel.nifti1.Nifti1Image): Template NIfTI image containing the affine matrix.
        scaled_vector (numpy.ndarray): The data array to be saved as a NIfTI file.
        saliency_output_file_name (str): Path where the compressed NIfTI file will be saved.
        
    Returns:
        None
    """
    try:
        # Ensure output directory exists and is writable
        output_dir = os.path.dirname(saliency_output_file_name)
        ensure_directory_exists(output_dir)
        
        # Check if output shapes match
        if img.shape != scaled_vector.shape:
            raise ValueError(f"Shape mismatch: template image shape {img.shape} != vector shape {scaled_vector.shape}")
        
        # Create NIfTI image with the scaled vector data and the template's affine matrix
        affine = img.affine
        array_img = nib.Nifti1Image(scaled_vector, affine)
        
        # Save directly as compressed file
        try:
            # NiBabel automatically compresses when filename ends with .gz
            nib.save(array_img, saliency_output_file_name)
        except Exception as save_error:
            raise IOError(f"Failed to save compressed NIfTI file: {str(save_error)}")
    except Exception as e:
        print(f"Error in create_nifti: {str(e)}")
        raise

def create_saliency_nifti(nifti_input, saliency_output_file_path, model_file_path, scaling_factor=256):
    """
    Create a saliency map NIfTI file from an input NIfTI file using a pre-trained model.
    
    This function loads the input image, computes its saliency map, normalizes and scales
    the saliency values, and saves the result as a compressed NIfTI file.
    
    Parameters:
        nifti_input (str): Path to the input NIfTI file.
        saliency_output_file_path (str): Path where the saliency map will be saved.
        model_file_path (str): Path to the pre-trained model weights file.
        scaling_factor (int, optional): Value to scale the normalized saliency map. 
                                      Higher values make the saliency map more visible.
                                      Default is 256 for 8-bit visualization.
        
    Returns:
        None
    """
    try:
        # Verify input file exists
        if not os.path.exists(nifti_input):
            raise FileNotFoundError(f"Input NIfTI file not found: {nifti_input}")
        
        # Load input image
        try:
            image = nib.load(nifti_input)
        except Exception as load_error:
            raise ValueError(f"Failed to load input NIfTI file: {str(load_error)}")
        
        # Ensure output directory exists and is writable
        salience_output_directory_path = os.path.dirname(saliency_output_file_path)
        ensure_directory_exists(salience_output_directory_path)
        
        # Compute saliency map
        array_data = compute_saliency(nifti_input, model_file_path)
        
        # Check array_data for valid values
        if np.all(array_data == 0):
            print("Warning: Saliency map contains all zeros")
        
        # Normalize and scale saliency values
        try:
            # L2 normalization of the saliency map
            # This is different from the min-max normalization used for the input image
            # L2 norm preserves the direction of gradients while standardizing their magnitude
            norm = np.linalg.norm(array_data)
            if norm == 0:
                print("Warning: Saliency norm is zero, using unnormalized data")
                normalized_vector = array_data
            else:
                normalized_vector = array_data / norm
                
            # Scale the normalized vector to a visible range
            # The scaling factor determines the intensity range in the output image
            # Default of 256 is suitable for 8-bit visualization tools
            scaled_data = np.dot(normalized_vector, scaling_factor)
            
            # Create output NIfTI file (directly as compressed file)
            create_nifti(image, scaled_data, saliency_output_file_path)
            
        except Exception as process_error:
            raise RuntimeError(f"Failed to process saliency map: {str(process_error)}")
    except Exception as e:
        print(f"Error in create_saliency_nifti: {str(e)}")
        raise

def main():
    """
    Main entry point for the script.
    
    Expects three command-line arguments:
    1. Path to the input NIfTI file
    2. Path where the saliency output will be saved
    3. Path to the pre-trained model weights file
    
    Optional arguments can be added in future versions:
    - Scaling factor for saliency map intensity
    - Output data type options
    - Customization of saliency calculation parameters
    
    Returns:
        int: 0 for success, 1 for failure
    """
    try:
        # Check if correct number of arguments are provided
        if len(sys.argv) != 4:
            print("Usage: python script.py <input_nifti> <output_path> <model_path>")
            print("  <input_nifti>: Path to the input NIfTI file (.nii or .nii.gz)")
            print("  <output_path>: Path where the saliency map will be saved (.nii.gz)")
            print("  <model_path>: Path to the pre-trained model weights file")
            return 1
            
        my_nifti_input = sys.argv[1]
        saliency_output_file_path = sys.argv[2]
        model_file_path = sys.argv[3]
        
        # Validate arguments
        is_valid, error_message = validate_args(
            my_nifti_input, 
            saliency_output_file_path, 
            model_file_path
        )
        
        if not is_valid:
            print(f"Error: {error_message}")
            return 1
        
        # Process the NIfTI file
        create_saliency_nifti(my_nifti_input, saliency_output_file_path, model_file_path)
        
        print(f"Saliency map successfully created: {saliency_output_file_path}")
        return 0
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())