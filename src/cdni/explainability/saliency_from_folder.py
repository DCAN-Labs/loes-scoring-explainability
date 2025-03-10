#!/usr/bin/env python3
"""
NIFTI Saliency Map Generator

This script processes NIFTI files to generate saliency maps using the CDNI explainability 
framework. It provides flexible file filtering options and comprehensive error handling.

Usage:
    python generate_saliency_maps.py input_folder output_folder model_path [options]

Author: Your Name
Date: March 10, 2025
"""

import sys
import os
import argparse
from tqdm import tqdm
import logging
import traceback
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import required modules with error handling
try:
    from cdni.explainability.saliency_visualization import create_saliency_nifti
except ImportError:
    logger.error("Could not import required modules. Make sure cdni package is installed.")
    sys.exit(1)


def create_saliency_maps(nifti_input_folder, saliency_output_folder, model_file_path, 
                         file_pattern=None, exclude_pattern=None, overwrite=True):
    """
    Create saliency maps for all NIFTI files in the input folder.
    
    This function processes NIFTI files (.nii or .nii.gz) from the input folder,
    applies the specified model to generate saliency maps, and saves the results
    to the output folder. It includes comprehensive error handling and file filtering.
    
    Args:
        nifti_input_folder (str): Path to folder containing input NIFTI files
        saliency_output_folder (str): Path to folder where output saliency maps will be saved
        model_file_path (str): Path to the model file used for saliency map generation
        file_pattern (str, optional): Regex pattern to include specific files
        exclude_pattern (str, optional): Regex pattern to exclude specific files
        overwrite (bool, optional): Whether to overwrite existing output files, defaults to True
        
    Returns:
        tuple: Counts of (processed, skipped, error) files
        
    Raises:
        FileNotFoundError: If input folder or model file doesn't exist
        PermissionError: If output folder can't be created or written to
        Exception: For other unexpected errors during processing
    """
    try:
        # --- Validation phase ---
        # Check if input folder exists and is accessible
        if not os.path.exists(nifti_input_folder):
            raise FileNotFoundError(f"Input folder not found: {nifti_input_folder}")
        if not os.path.isdir(nifti_input_folder):
            raise NotADirectoryError(f"Input path is not a directory: {nifti_input_folder}")
        
        # Check if model file exists and is accessible
        if not os.path.exists(model_file_path):
            raise FileNotFoundError(f"Model file not found: {model_file_path}")
        if not os.path.isfile(model_file_path):
            raise IsADirectoryError(f"Model path is not a file: {model_file_path}")
        
        # Create output folder if it doesn't exist
        if not os.path.exists(saliency_output_folder):
            logger.info(f"Creating output folder: {saliency_output_folder}")
            try:
                os.makedirs(saliency_output_folder)
            except PermissionError:
                raise PermissionError(f"Cannot create output directory: {saliency_output_folder} (permission denied)")
        elif not os.access(saliency_output_folder, os.W_OK):
            raise PermissionError(f"Output directory is not writable: {saliency_output_folder}")
        
        # --- File discovery and filtering phase ---
        # Get list of files
        logger.debug(f"Scanning directory: {nifti_input_folder}")
        files = os.listdir(nifti_input_folder)
        
        # First filter for NIFTI files (common extensions: .nii, .nii.gz)
        nifti_files = [f for f in files if f.endswith(('.nii', '.nii.gz'))]
        logger.debug(f"Found {len(nifti_files)} NIFTI files out of {len(files)} total files")
        
        # Apply custom include pattern if provided
        if file_pattern:
            try:
                pattern = re.compile(file_pattern)
                matched_files = [f for f in nifti_files if pattern.search(f)]
                logger.info(f"Applied include pattern '{file_pattern}': {len(matched_files)} files matched")
                nifti_files = matched_files
            except re.error as e:
                logger.error(f"Invalid include pattern '{file_pattern}': {str(e)}")
        
        # Apply custom exclude pattern if provided
        if exclude_pattern:
            try:
                pattern = re.compile(exclude_pattern)
                filtered_files = [f for f in nifti_files if not pattern.search(f)]
                logger.info(f"Applied exclude pattern '{exclude_pattern}': {len(nifti_files) - len(filtered_files)} files excluded")
                nifti_files = filtered_files
            except re.error as e:
                logger.error(f"Invalid exclude pattern '{exclude_pattern}': {str(e)}")
        
        # Check if any files remain after filtering
        if not nifti_files:
            logger.warning(f"No matching NIFTI files found in {nifti_input_folder}")
            return (0, 0, 0)
        
        logger.info(f"Processing {len(nifti_files)} NIFTI files")
        
        # --- Processing phase ---
        # Initialize counters for statistics
        processed_count = 0
        skipped_count = 0
        error_count = 0
        
        # Process each file with progress bar
        for file in tqdm(nifti_files, desc="Generating saliency maps", unit="file"):
            try:
                saliency_output_file_path = os.path.join(saliency_output_folder, file)
                saliency_input_file_path = os.path.join(nifti_input_folder, file)
                
                # Check if input file exists and is readable
                if not os.path.isfile(saliency_input_file_path):
                    logger.error(f"Input file not found: {saliency_input_file_path}")
                    skipped_count += 1
                    continue
                
                # Check if file is readable and has minimum size
                try:
                    file_size = os.path.getsize(saliency_input_file_path)
                    if file_size == 0:
                        logger.warning(f"Skipping empty file: {saliency_input_file_path}")
                        skipped_count += 1
                        continue
                except OSError as e:
                    logger.error(f"Cannot access file {saliency_input_file_path}: {str(e)}")
                    skipped_count += 1
                    continue
                
                # Check if output file already exists
                if os.path.exists(saliency_output_file_path):
                    if overwrite:
                        logger.debug(f"Output file already exists, overwriting: {saliency_output_file_path}")
                    else:
                        logger.info(f"Skipping existing output file: {saliency_output_file_path}")
                        skipped_count += 1
                        continue
                
                # Generate saliency map using the CDNI library function
                logger.debug(f"Processing file: {file}")
                create_saliency_nifti(saliency_input_file_path, saliency_output_file_path, model_file_path)
                logger.debug(f"Successfully created saliency map for {file}")
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Error processing file {file}: {str(e)}")
                logger.debug(traceback.format_exc())
                error_count += 1
                continue
        
        # --- Report phase ---
        # Generate and log processing summary
        logger.info(f"Processing summary: {processed_count} successful, {skipped_count} skipped, {error_count} failed")
        return (processed_count, skipped_count, error_count)
                
    except Exception as e:
        logger.error(f"Failed to create saliency maps: {str(e)}")
        logger.debug(traceback.format_exc())
        raise


def main():
    """
    Main entry point for the script.
    
    Parses command line arguments, sets up logging, and calls the 
    create_saliency_maps function with appropriate parameters.
    
    Returns:
        int: Return code (0 for success, 1 for error)
    """
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(
            description='Create saliency maps for NIFTI files using CDNI explainability',
            epilog='Example: python %(prog)s /data/nifti /data/saliency /models/my_model.h5 -v'
        )
        parser.add_argument('input_folder', 
                          help='Folder containing input NIFTI files')
        parser.add_argument('output_folder', 
                          help='Folder where output saliency maps will be saved')
        parser.add_argument('model_path', 
                          help='Path to the model file used for saliency map generation')
        parser.add_argument('--pattern', '-p', 
                          help='Regex pattern to include specific files (e.g., "sub-\\d+_T1")')
        parser.add_argument('--exclude', '-e', 
                          help='Regex pattern to exclude specific files (e.g., "test_")')
        parser.add_argument('--recursive', '-r', 
                          action='store_true', 
                          help='Process files in subdirectories recursively (not implemented yet)')
        parser.add_argument('--no-overwrite', '-n', 
                          action='store_true',
                          help='Do not overwrite existing output files')
        parser.add_argument('--verbose', '-v', 
                          action='store_true', 
                          help='Enable verbose logging')
        parser.add_argument('--version', 
                          action='version', 
                          version='%(prog)s 1.0.0')
        
        args = parser.parse_args()
        
        # Set debug level if verbose flag is used
        if args.verbose:
            logger.setLevel(logging.DEBUG)
            logger.debug("Verbose logging enabled")
        
        # Run the main function
        if args.recursive:
            # TODO: Implement recursive processing (not implemented in this update)
            logger.warning("Recursive processing not implemented yet, ignoring --recursive flag")
            
        logger.info("Starting saliency map generation")
        processed, skipped, errors = create_saliency_maps(
            args.input_folder, 
            args.output_folder, 
            args.model_path,
            file_pattern=args.pattern,
            exclude_pattern=args.exclude,
            overwrite=not args.no_overwrite
        )
        
        # Report final status
        if errors > 0:
            logger.warning(f"Processing completed with {errors} errors")
        else:
            logger.info("Processing completed successfully")
        
        return 0 if errors == 0 else 1
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        logger.debug(traceback.format_exc())
        return 1


# Script entry point
if __name__ == '__main__':
    sys.exit(main())