"""
Segmentation Module

This module provides functionality for segmenting medical images using watershed
algorithm to separate connected anatomical structures in CT scan data.

Author: Amir Bhattarai
Date: June 07, 2025
Version: 1.0
"""

# Standard library imports
import os
from typing import Tuple, Optional

# Third-party imports
import nibabel as nib
import torch
import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import distance_transform_edt
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

# Local imports
from plots import visualize_segments


def watershed_segmentation(volume_3d: np.ndarray, 
                          footprint_size: Tuple[int, int, int] = (100, 100, 100)) -> np.ndarray:
    """
    Perform watershed segmentation on a 3D medical image volume.
    
    This function uses distance transform and watershed algorithm to separate
    connected anatomical structures, particularly effective for bone segmentation
    in CT scans where femur and tibia may appear connected.
    
    Args:
        volume_3d (np.ndarray): 3D binary volume data to be segmented
        footprint_size (Tuple[int, int, int]): Size of the neighborhood for 
                                             peak detection. Default: (100, 100, 100)
    
    Returns:
        np.ndarray: Labeled 3D volume where each connected component has a unique label
        
    Raises:
        ValueError: If input volume is not 3D
        TypeError: If input volume is not numpy array
        
    Example:
        >>> volume = np.random.randint(0, 2, (50, 50, 50))
        >>> labels = watershed_segmentation(volume)
        >>> print(f"Found {np.max(labels)} segments")
    """
    if not isinstance(volume_3d, np.ndarray):
        raise TypeError("Input volume must be a numpy array")
        
    if volume_3d.ndim != 3:
        raise ValueError("Input volume must be 3D")
    
    # Convert to boolean for distance transform
    binary_volume = volume_3d.astype(bool)
    
    # Calculate Euclidean distance transform
    # Distance from each background pixel to nearest foreground pixel
    distance = distance_transform_edt(binary_volume)
    
    # Find local maxima in distance map as watershed markers
    # These represent centers of objects to be separated
    coords = peak_local_max(
        distance, 
        footprint=np.ones(footprint_size), 
        labels=binary_volume
    )
    
    # Create marker mask from peak coordinates
    marker_mask = np.zeros(distance.shape, dtype=bool)
    marker_mask[tuple(coords.T)] = True
    
    # Label connected components in marker mask
    markers, _ = ndi.label(marker_mask)
    
    # Apply watershed algorithm using negative distance as elevation map
    # Negative distance ensures watershed flows from peaks (high distance) to valleys
    labels = watershed(-distance, markers, mask=binary_volume)
    
    return labels

def create_tensor(bg_mask:np.ndarray, labels: np.ndarray, name: str = "tensor_3d") -> torch.Tensor:
    """
    
    Create a 3D tensor that contains RGB color information for labels.

    Args:
        
        labels (np.ndarray): Segmentation result

    Returns:
        torch.Tensor: 3D tensor with shape [channels, slices, height, width]

    """
    
    # Color coding: background (white), tibia (green), femur (red)
    color_map = {
        0: [128, 128, 128],  # background
        1: [0, 255, 0],      # tibia
        2: [255, 0, 0],      # femur
    }

    # Create an empty array to store color information
    volume_rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)

    if name == "tibia":
        volume_rgb[labels == 1] = color_map[1]
    
    elif name == "femur":
        volume_rgb[labels == 2] = color_map[2]

    elif name == "background":
        volume_rgb[bg_mask == 128] = color_map[0]

    else:
        volume_rgb[bg_mask == 128] = color_map[0]
        volume_rgb[labels == 1] = color_map[1]
        volume_rgb[labels == 2] = color_map[2]

    # Normalize the values between 0. and 1.
    volume_rgb = volume_rgb.astype(np.float32) / 255.0

    # Create and reshape a tensor
    tensor_3d = torch.from_numpy(volume_rgb).permute(3, 2, 0, 1)

    return tensor_3d


def load_medical_volume(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load bone mask volume from NIfTI file.
    
    Args:
        file_path (str): Path to the .nii or .nii.gz file
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Volume data and spacing information
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file cannot be loaded
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    try:
        volume = nib.load(file_path)
        data = volume.get_fdata()
        spacing = volume.header.get_zooms()
        return data, spacing
    except Exception as e:
        raise ValueError(f"Error loading file {file_path}: {str(e)}")


def main() -> None:
    """
    Main execution function for segmentation module.
    
    Loads mask data from environment variable and performs watershed segmentation,
    then visualizes the results.
    """
    try:
        # Get bone mask data path from environment variable
        mask_path = os.environ.get("DATASET_PATH")+"3702_left_knee_bone_mask.nii.gz"
        if not mask_path:
            raise ValueError("DATASET_PATH environment variable not set")
        
        # Load bone mask volume
        mask_data, _ = load_medical_volume(mask_path)

        # Get background mask data path from environment variable
        bg_path = os.environ.get("DATASET_PATH")+"3702_left_knee_bg_mask.nii.gz"
        if not bg_path:
            raise ValueError("DATASET_PATH environment variable not set")
        
        # Load background mask volume
        bg_mask, _ = load_medical_volume(bg_path)
        
        # Stack 2D slices into 3D volume if needed
        if mask_data.ndim == 2:
            volume_3d = np.expand_dims(mask_data, axis=0)
        else:
            volume_3d = mask_data
        
        # Perform watershed segmentation
        print("Performing watershed segmentation...")
        labels = watershed_segmentation(volume_3d)

        # Define region types for tensor creation and storage
        regions = ["tibia", "femur", "background", "tensor"]

        # Create and save PyTorch tensor objects for each anatomical region
        for region in regions:
            print(f"Creating and saving the {region}_3d.pth tensor")
            
            # Generate 3D tensor for current region
            tensor_3d = create_tensor(bg_mask, labels, region)
            
            # Construct output filepath and save tensor
            output_path = os.environ.get("STORE_LOCATION") + f"{region}_3d.pth"
            torch.save(tensor_3d, output_path)

        # Display results
        num_segments = np.max(labels)
        print(f"Segmentation complete. Found {num_segments} segments.")
        
        # Visualize segmentation results
        visualize_segments(labels)
        
    except Exception as e:
        print(f"Error in segmentation: {str(e)}")
        raise


if __name__ == "__main__":
    main()