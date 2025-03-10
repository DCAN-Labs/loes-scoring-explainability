import sys
import os
import argparse
from tqdm import tqdm
import logging
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from cdni.explainability.saliency_visualization import create_saliency_nifti
except ImportError:
    logger.error("Could not import required modules. Make sure cdni package is installed.")
    sys.exit(1)

def create_saliency_maps(nifti_input_folder, saliency_output_folder, model_file_path):
    """
    Create saliency maps for all NIFTI files in the input folder.
    
    Args:
        nifti_input_folder (str): Path to folder containing input NIFTI files
        saliency_output_folder (str): Path to folder where output saliency maps will be saved
        model_file_path (str): Path to the model file
    """
    try:
        # Check if input folder exists
        if not os.path.exists(nifti_input_folder):
            raise FileNotFoundError(f"Input folder not found: {nifti_input_folder}")
        
        # Check if model file exists
        if not os.path.exists(model_file_path):
            raise FileNotFoundError(f"Model file not found: {model_file_path}")
        
        # Create output folder if it doesn't exist
        if not os.path.exists(saliency_output_folder):
            logger.info(f"Creating output folder: {saliency_output_folder}")
            os.makedirs(saliency_output_folder)
        
        # Get list of files
        files = os.listdir(nifti_input_folder)
        # Filter for NIFTI files (common extensions: .nii, .nii.gz)
        nifti_files = [f for f in files if f.endswith(('.nii', '.nii.gz'))]
        
        if not nifti_files:
            logger.warning(f"No NIFTI files found in {nifti_input_folder}")
            return
        
        logger.info(f"Processing {len(nifti_files)} NIFTI files")
        
        # Process each file
        for file in tqdm(nifti_files):
            try:
                saliency_output_file_path = os.path.join(saliency_output_folder, file)
                saliency_input_file_path = os.path.join(nifti_input_folder, file)
                
                # Check if input file exists and is readable
                if not os.path.isfile(saliency_input_file_path):
                    logger.error(f"Input file not found: {saliency_input_file_path}")
                    continue
                
                # Check if output file already exists (optional)
                if os.path.exists(saliency_output_file_path):
                    logger.warning(f"Output file already exists, overwriting: {saliency_output_file_path}")
                
                # Create saliency map
                create_saliency_nifti(saliency_input_file_path, saliency_output_file_path, model_file_path)
                logger.debug(f"Successfully created saliency map for {file}")
                
            except Exception as e:
                logger.error(f"Error processing file {file}: {str(e)}")
                logger.debug(traceback.format_exc())
                continue
                
    except Exception as e:
        logger.error(f"Failed to create saliency maps: {str(e)}")
        logger.debug(traceback.format_exc())
        raise

if __name__ == '__main__':
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Create saliency maps for NIFTI files')
        parser.add_argument('input_folder', help='Folder containing input NIFTI files')
        parser.add_argument('output_folder', help='Folder where output saliency maps will be saved')
        parser.add_argument('model_path', help='Path to the model file')
        parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
        
        args = parser.parse_args()
        
        # Set debug level if verbose flag is used
        if args.verbose:
            logger.setLevel(logging.DEBUG)
        
        # Run the main function
        create_saliency_maps(args.input_folder, args.output_folder, args.model_path)
        logger.info("Processing completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        logger.debug(traceback.format_exc())
        sys.exit(1)