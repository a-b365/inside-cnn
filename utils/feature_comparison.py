#!/usr/bin/env python3
"""
Feature Comparison Module for Different Knee Regions

This module provides functionality for comparing extracted features from different
anatomical regions using cosine similarity. It's designed for analyzing relationships
between different anatomical structures (tibia, femur, background) in medical images.

The module computes pairwise similarity scores between regions across multiple
neural network layers and exports results to CSV format for further analysis.

Author: Amir Bhattarai
Date: June 10, 2025
Version: 1.0
"""

import os
from typing import Dict, List, Tuple, Any

import torch
import pandas as pd
import torch.nn.functional as F

from feature_extraction import (
    extract_gap_features, 
    load_pretrained_model, 
    load_tensor_data
)


def load_anatomical_tensors(tensor_names: List[str], store_location: str = None) -> Dict[str, torch.Tensor]:
    """
    Load multiple anatomical region tensors from files.
    
    Args:
        tensor_names: List of tensor file names to load
        store_location: Base directory path (uses STORE_LOCATION env var if None)
        
    Returns:
        Dict mapping region names to loaded tensors
        
    Raises:
        FileNotFoundError: If any tensor file doesn't exist
        RuntimeError: If tensor loading fails
    """
    tensors = {}
    
    for tensor_name in tensor_names:
        # Extract region name from filename (remove _3d.pth suffix)
        region_name = tensor_name.replace("_3d.pth", "")
        
        try:
            print(f"Loading {region_name} tensor...")
            tensors[region_name] = load_tensor_data(tensor_name, store_location)
        except Exception as e:
            print(f"Failed to load {tensor_name}: {e}")
            raise
    
    return tensors


def extract_features_for_regions(
    model: torch.nn.Module,
    region_tensors: Dict[str, torch.Tensor],
    layer_names: List[str]
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Extract features from all anatomical regions for specified layers.
    
    Args:
        model: The neural network model
        region_tensors: Dict mapping region names to tensors
        layer_names: List of layer names to extract features from
        
    Returns:
        Nested dict: {region_name: {layer_name: feature_tensor}}
    """
    print("Extracting features for all regions...")
    feature_lookup = {}
    
    for region_name, tensor in region_tensors.items():
        print(f"  Processing {region_name}...")
        region_features = extract_gap_features(model, tensor, layer_names)
        feature_lookup[region_name] = region_features
    
    return feature_lookup


def compute_cosine_similarity(
    feature1: torch.Tensor, 
    feature2: torch.Tensor
) -> float:
    """
    Compute cosine similarity between two feature vectors.
    
    Args:
        feature1: First feature tensor
        feature2: Second feature tensor
        
    Returns:
        float: Cosine similarity score between -1 and 1
    """
    similarity = F.cosine_similarity(feature1, feature2, dim=1)
    return similarity.item()


def create_similarity_comparison_data(
    feature_lookup: Dict[str, Dict[str, torch.Tensor]],
    region_pairs: List[Tuple[str, str]],
    layer_mapping: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
    Create comparison data by computing similarities between region pairs.
    
    Args:
        feature_lookup: Features for each region and layer
        region_pairs: List of (region1, region2) tuples to compare
        layer_mapping: Dict mapping layer positions to layer names
        
    Returns:
        List of dictionaries containing comparison data
    """
    comparison_data = []
    
    print("Computing pairwise similarities...")
    
    for region1, region2 in region_pairs:
        print(f"  Comparing {region1} <-> {region2}")
        
        row_data = {"Image Pair": f"{region1} <-> {region2}"}
        
        # Compute similarity for each layer
        for position, layer_name in layer_mapping.items():
            try:
                feature1 = feature_lookup[region1][layer_name]
                feature2 = feature_lookup[region2][layer_name]
                
                similarity = compute_cosine_similarity(feature1, feature2)
                row_data[position] = similarity
                
            except KeyError as e:
                print(f"    Warning: Missing feature for {e}")
                row_data[position] = None
        
        comparison_data.append(row_data)
    
    return comparison_data


def save_similarity_results(
    comparison_data: List[Dict[str, Any]], 
    output_path: str
) -> None:
    """
    Save similarity comparison results to CSV file.
    
    Args:
        comparison_data: List of comparison result dictionaries
        output_path: Path where CSV file should be saved
        
    Raises:
        OSError: If file cannot be written
    """
    try:
        # Create DataFrame and save to CSV
        df = pd.DataFrame(comparison_data)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        df.to_csv(output_path, index=False)
        print(f"Similarity results saved to: {output_path}")
        
        # Display preview of results
        print("\nSimilarity Results Preview:")
        print("-" * 60)
        print(df.to_string(index=False, float_format='{:.4f}'.format))
        
    except Exception as e:
        raise OSError(f"Failed to save results to {output_path}: {e}")


def validate_regions_and_layers(
    region_tensors: Dict[str, torch.Tensor],
    region_pairs: List[Tuple[str, str]],
    layer_names: List[str]
) -> None:
    """
    Validate that all required regions exist and layer names are provided.
    
    Args:
        region_tensors: Available region tensors
        region_pairs: Region pairs to be compared
        layer_names: Layer names for feature extraction
        
    Raises:
        ValueError: If validation fails
    """
    # Check that all regions in pairs exist
    required_regions = set()
    for r1, r2 in region_pairs:
        required_regions.update([r1, r2])
    
    missing_regions = required_regions - set(region_tensors.keys())
    if missing_regions:
        raise ValueError(f"Missing region tensors: {missing_regions}")
    
    # Check that layer names are provided
    if not layer_names:
        raise ValueError("No layer names provided for feature extraction")
    
    print(f"Validation passed: {len(region_tensors)} regions, {len(layer_names)} layers")


def run_feature_comparison_pipeline() -> None:
    """
    Execute the complete feature comparison pipeline.
    
    This function orchestrates the entire process of loading models and data,
    extracting features, computing similarities, and saving results.
    """
    try:
        # Configuration
        model_filename = "model_3d.pth"
        tensor_filenames = ["tibia_3d.pth", "femur_3d.pth", "background_3d.pth"]
        output_filename = "sim-score.csv"
        
        # Define anatomical region pairs for comparison
        region_pairs = [
            ("tibia", "femur"),
            ("tibia", "background"),
            ("femur", "background")
        ]
        
        # Define layers to extract features from (DenseNet121 final block)
        layer_mapping = {
            "last": "features.denseblock4.denselayer16.conv2",
            "third-last": "features.denseblock4.denselayer15.conv2",
            "fifth-last": "features.denseblock4.denselayer14.conv2"
        }
        
        layer_names = list(layer_mapping.values())
        
        # Get store location
        store_location = os.getenv("STORE_LOCATION")
        if store_location is None:
            raise EnvironmentError("STORE_LOCATION environment variable not set")
        
        output_path = os.path.join(store_location, output_filename)
        
        # Step 1: Load model and tensors
        print("Step 1: Loading model and anatomical tensors...")
        model_3d = load_pretrained_model(model_filename)
        region_tensors = load_anatomical_tensors(tensor_filenames)
        
        # Step 2: Validate inputs
        print("\nStep 2: Validating inputs...")
        validate_regions_and_layers(region_tensors, region_pairs, layer_names)
        
        # Step 3: Extract features for all regions
        print("\nStep 3: Extracting features...")
        feature_lookup = extract_features_for_regions(model_3d, region_tensors, layer_names)
        
        # Step 4: Compute pairwise similarities
        print("\nStep 4: Computing similarities...")
        comparison_data = create_similarity_comparison_data(
            feature_lookup, region_pairs, layer_mapping
        )
        
        # Step 5: Save results
        print("\nStep 5: Saving results...")
        save_similarity_results(comparison_data, output_path)
        
        print("\nFeature comparison pipeline completed successfully!")
        
    except Exception as e:
        print(f"Error in feature comparison pipeline: {e}")
        raise


def main() -> None:
    """
    Main function to execute the feature comparison analysis.
    
    Loads anatomical region data, extracts deep features, computes similarity
    scores between different regions, and saves results for analysis.
    """
    print("Starting Feature Comparison Analysis")
    print("=" * 50)
    
    run_feature_comparison_pipeline()


if __name__ == "__main__":
    main()