"""
Mask Generation Pipeline

This module provides a comprehensive pipeline for segmenting knee region from
CT volumes. The segmentation process uses thresholding, morphological
operations, and watershed segmentation to create accurate binary masks.

The pipeline is specifically designed for knee anatomy and uses different parameters
for different anatomical regions along the slice axis to account for varying tissue
densities and structures.

Author: Amir Bhattarai
Date: June 11, 2025
Version: 1.0
"""

import os
from typing import Tuple, Dict, Any

import cv2
import numpy as np
import nibabel as nib
from skimage.filters import gaussian
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects, binary_opening, disk
from scipy.ndimage import binary_fill_holes, zoom, distance_transform_edt, label


def load_volume(volume_path: str) -> Tuple[nib.Nifti1Image, np.ndarray]:
    """
    Load a medical image volume from a NIfTI file.
    
    Args:
        volume_path: Path to the NIfTI volume file
        
    Returns:
        Tuple containing the NIfTI image object and the data array
        
    Raises:
        FileNotFoundError: If the volume file doesn't exist
        ValueError: If the volume cannot be loaded
    """
    if not os.path.exists(volume_path):
        raise FileNotFoundError(f"Volume file not found: {volume_path}")
    
    try:
        volume = nib.load(volume_path)
        data = volume.get_fdata()
        return volume, data
    except Exception as e:
        raise ValueError(f"Failed to load volume: {e}")


def remove_disconnected_blobs(clean_mask: np.ndarray) -> np.ndarray:
    """
    Remove disconnected blobs and keep only the largest connected component.
    
    Uses distance transform and watershed segmentation to separate loosely
    connected regions, then keeps only the largest component.
    
    Args:
        clean_mask: Binary mask to process
        
    Returns:
        np.ndarray: Binary mask with only the largest connected component
    """
    # Compute distance transform for watershed
    distance = distance_transform_edt(clean_mask)
    
    # Find local maxima as watershed markers
    coords = peak_local_max(
        distance, 
        labels=clean_mask, 
        footprint=np.ones((10, 10)), 
        min_distance=25
    )
    
    # Create markers from peak coordinates
    markers = np.zeros_like(distance, dtype=bool)
    markers[tuple(coords.T)] = True
    markers, _ = label(markers)
    
    # Apply watershed segmentation
    labels_ws = watershed(-distance, markers, mask=clean_mask)
    
    # Keep only the largest region
    output_mask = np.zeros_like(clean_mask, dtype=bool)
    regions = regionprops(labels_ws)
    
    if regions:
        largest_region = max(regions, key=lambda r: r.area)
        output_mask[labels_ws == largest_region.label] = True
    
    return output_mask


def create_slice_mask(
    slice_data: np.ndarray,
    threshold_value: int = 240,
    min_size: int = 800,
    zoom_factor: float = 4.0,
    sigma: float = 2.0
) -> np.ndarray:
    """
    Create a binary mask for a single slice of the volume.
    
    This function applies upsampling, Gaussian filtering, thresholding,
    and morphological operations to create a clean binary mask.
    
    Args:
        slice_data: 2D array representing a single slice
        threshold_value: Intensity threshold for binarization
        min_size: Minimum size of objects to keep after cleaning
        zoom_factor: Factor for upsampling before processing
        sigma: Standard deviation for Gaussian blur
        
    Returns:
        np.ndarray: Binary mask for the input slice
    """
    # Upsample to improve resolution for processing
    upsampled = zoom(slice_data, (zoom_factor, zoom_factor), order=0)
    
    # Apply Gaussian blur to reduce noise
    blurred = gaussian(upsampled, sigma=sigma)
    
    # Apply threshold to create binary mask
    _, thresh = cv2.threshold(
        blurred, 
        threshold_value, 
        255, 
        cv2.THRESH_BINARY
    )
    
    # Downsample back to original resolution
    downsampled = zoom(thresh, (1/zoom_factor, 1/zoom_factor), order=0)
    
    # Fill internal holes
    filled_mask = binary_fill_holes(downsampled)
    
    # Apply morphological opening to remove small artifacts
    opened_mask = binary_opening(filled_mask, disk(1))
    
    # Remove small connected components
    clean_mask = remove_small_objects(
        opened_mask.astype(bool), 
        min_size=min_size
    )
    
    return clean_mask


def process_slice_range(
    data: np.ndarray,
    binary_mask: np.ndarray,
    start_slice: int,
    end_slice: int,
    threshold_value: int,
    min_size: int,
    apply_blob_removal: bool = False
) -> None:
    """
    Process a range of slices with specified parameters.
    
    Args:
        data: 3D volume data array
        binary_mask: 3D binary mask array to store results
        start_slice: Starting slice index (inclusive)
        end_slice: Ending slice index (exclusive)
        threshold_value: Intensity threshold for binarization
        min_size: Minimum size of objects to keep
        apply_blob_removal: Whether to apply blob removal post-processing
    """
    for i in range(start_slice, end_slice):
        # Create mask for current slice
        clean_mask = create_slice_mask(
            data[:, :, i], 
            threshold_value, 
            min_size
        )
        
        # Apply blob removal if specified
        if apply_blob_removal:
            clean_mask = remove_disconnected_blobs(clean_mask)
        
        # Store mask in the binary volume
        binary_mask[:, :, i] = clean_mask.astype(np.uint8)


def run_knee_segmentation(data: np.ndarray) -> np.ndarray:
    """
    Execute the complete knee segmentation pipeline on volume data.
    
    This function processes the entire volume using different parameters
    for different anatomical regions to optimize segmentation quality.
    The parameters are tuned for typical knee anatomy in CT/MRI scans.
    
    Args:
        data: 3D volume data array
        
    Returns:
        np.ndarray: 3D binary mask array
    """
    print("Starting knee segmentation pipeline...")
    
    # Initialize binary mask
    binary_mask = np.zeros_like(data, dtype=np.uint8)
    
    # Define segmentation parameters for different anatomical regions
    segmentation_params = [
        # (start, end, threshold, min_size, apply_blob_removal)
        (0, 90, 240, 800, False),      # Upper region
        (90, 94, 160, 800, True),      # Transition region with blob removal
        (94, 103, 180, 800, False),    # Mid-upper region
        (103, 104, 280, 800, False),   # Narrow transition
        (104, 108, 130, 60, False),    # Lower threshold, small objects
        (108, 130, 180, 1200, False),  # Mid-lower region
        (130, 216, 240, 800, False),   # Lower region
    ]
    
    # Process each region with its specific parameters
    for i, (start, end, threshold, min_size, blob_removal) in enumerate(segmentation_params):
        print(f"Processing region {i+1}/7: slices {start}-{end-1}")
        process_slice_range(
            data, binary_mask, start, end, threshold, min_size, blob_removal
        )
    
    print("Segmentation pipeline completed successfully!")
    return binary_mask


def save_segmentation_mask(
    binary_mask: np.ndarray, 
    volume: nib.Nifti1Image, 
    output_path: str
) -> None:
    """
    Save the binary mask as a NIfTI file.
    
    Args:
        binary_mask: 3D binary mask array
        volume: Original NIfTI image object (for affine transformation)
        output_path: Path where the mask should be saved
        
    Raises:
        OSError: If there are issues with file writing
    """
    try:
        # Create NIfTI image with same affine transformation as original
        nifti_img = nib.Nifti1Image(binary_mask, affine=volume.affine)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the mask
        nib.save(nifti_img, output_path)
        print(f"Segmentation mask saved to: {output_path}")
        
    except Exception as e:
        raise OSError(f"Failed to save mask: {e}")


def get_segmentation_statistics(binary_mask: np.ndarray) -> Dict[str, Any]:
    """
    Calculate basic statistics about the segmentation result.
    
    Args:
        binary_mask: 3D binary mask array
        
    Returns:
        Dict containing statistics including volume, slice coverage, etc.
    """
    non_zero_slices = np.sum(np.any(binary_mask, axis=(0, 1)))
    total_volume = np.sum(binary_mask)
    
    stats = {
        'total_voxels': int(total_volume),
        'non_zero_slices': int(non_zero_slices),
        'total_slices': binary_mask.shape[2],
        'volume_shape': binary_mask.shape
    }
    
    return stats


def print_segmentation_statistics(stats: Dict[str, Any]) -> None:
    """
    Print segmentation statistics in a formatted way.
    
    Args:
        stats: Dictionary containing segmentation statistics
    """
    print("\nSegmentation Statistics:")
    print(f"  Total segmented voxels: {stats['total_voxels']:,}")
    print(f"  Slices with segmentation: {stats['non_zero_slices']}/{stats['total_slices']}")
    print(f"  Volume dimensions: {stats['volume_shape']}")


def main() -> None:
    """
    Main function to execute the knee segmentation pipeline.
    
    Loads the volume from the path specified in DATASET_PATH environment variable,
    runs the segmentation pipeline, and saves the result.
    """
    # Get dataset path from environment variable
    dataset_path = os.getenv("DATASET_PATH")
    if dataset_path is None:
        raise EnvironmentError(
            "DATASET_PATH environment variable is not set. "
            "Please set it to specify the dataset location."
        )
    
    # Define input and output paths
    input_filename = "3702_left_knee.nii.gz"
    output_filename = "3702_left_knee_bone_mask.nii.gz"
    
    input_path = os.path.join(dataset_path, input_filename)
    output_path = os.path.join(dataset_path, output_filename)
    
    try:
        # Load the medical image volume
        print(f"Loading volume from: {input_path}")
        volume, data = load_volume(input_path)
        
        # Run segmentation pipeline
        binary_mask = run_knee_segmentation(data)
        
        # Save the segmentation results
        save_segmentation_mask(binary_mask, volume, output_path)
        
        # Calculate and print statistics
        stats = get_segmentation_statistics(binary_mask)
        print_segmentation_statistics(stats)
        
    except Exception as e:
        print(f"Error during segmentation: {e}")
        raise


if __name__ == "__main__":
    main()