import os

import cv2
import numpy as np
import nibabel as nib

from skimage.filters import gaussian
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects, binary_opening, disk
from scipy.ndimage import binary_fill_holes, zoom, distance_transform_edt, label


# Segmentation
volume = nib.load(os.getenv("DATASET_PATH")+"3702_left_knee.nii.gz")

data = volume.get_fdata()

binary_mask = np.zeros_like(data)

def remove_blobs(clean_mask):

    # Distance transform and peak detection
    distance = distance_transform_edt(clean_mask)
    coords = peak_local_max(distance, labels=clean_mask, footprint=np.ones((10, 10)), min_distance=25)
    
    # Label the peaks
    markers = np.zeros_like(distance, dtype=bool)
    markers[tuple(coords.T)] = True
    markers, _ = label(markers)

    # Watershed to separate loosely connected regions
    labels_ws = watershed(-distance, markers, mask=clean_mask)

    output_mask = np.zeros_like(clean_mask, dtype=bool)
    for region in sorted(regionprops(labels_ws), key=lambda r: r.area, reverse=True)[:1]:
        output_mask[labels_ws == region.label] = True

    return output_mask
    

def mask_slice(slice, threshold_value=240, min_size=800, zoom_factor=4, sigma=2):
    """
    Create a binary mask for a specific slice index in the CT volume.

    Parameters:
        slice (np.ndarray): Slice  to process.
        threshold_value (int): Intensity threshold for binarization.
        sigma (float): Gaussian blur sigma value.
        zoom_factor (int or float): Factor to zoom in before processing.

    Returns:
        np.ndarray: A binary mask for the given slice.
    """

    # Upsample to improve resolution for thresholding
    zoom_in = zoom(slice, (zoom_factor, zoom_factor), order=0)

    # Apply Gaussian blur to reduce noise
    blurred = gaussian(zoom_in, sigma=sigma)

    # Thresholding to create binary mask
    _, thresh = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)

    # Downsample to original shape
    zoom_out = zoom(thresh, (1/zoom_factor, 1/zoom_factor), order=0)

    # Fill internal holes
    filled_mask = binary_fill_holes(zoom_out)

    # Optional: Apply morphological opening to remove small artifacts
    opening = binary_opening(filled_mask, disk(1), mode="ignore")

    # Remove small objects to clean up the mask
    clean_mask = remove_small_objects(opening.astype(bool), min_size=min_size)

    return clean_mask

# Define the range of slices to process
def process_slices(start, end, threshold_value, min_size):

    for i in range(start, end):
        
        # Generate mask from the slice
        clean_mask = mask_slice(data[:, :, i], threshold_value, min_size)

        if i in range(90, 94):

            clean_mask = remove_blobs(clean_mask)

        # Store the mask in the binary volume
        binary_mask[:, :, i] = clean_mask.astype(np.uint8)
    

if __name__ == "__main__":

    # Extract the slice
    process_slices(0, 90, 240, 800)
    process_slices(90, 94, 160, 800)
    process_slices(94, 103, 180, 800)
    process_slices(103, 104, 280, 800)
    process_slices(104, 108, 130, 60)
    process_slices(108, 130, 180, 1200)
    process_slices(130, 216, 240, 800)

    # Save the segmented mask as a NIfTI file using the mask volume affine matrix
    nifti_img = nib.Nifti1Image(binary_mask, affine=volume.affine)
    nib.save(nifti_img, os.getenv("DATASET_PATH")+"3702_left_knee_mask_final.nii.gz")
    print(f"Mask saved to {os.getenv("DATASET_PATH")}")

    


