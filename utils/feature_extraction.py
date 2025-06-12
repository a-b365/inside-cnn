#!/usr/bin/env python3
"""
Feature Extraction Module for Medical Image Analysis

This module provides utilities for extracting deep features from 3D neural network models
using forward hooks. It's specifically designed for medical image analysis where feature
extraction from intermediate layers is needed for comparison and analysis.

The module supports Global Average Pooling (GAP) feature extraction from specified
layers of pre-trained models, particularly useful for analyzing anatomical structures
in medical imaging.

Author: Amir Bhattarai
Date: June 09, 2025
Version: 1.0
"""

import os
from typing import Dict, List, Tuple, Callable, Any

import torch
import torch.nn.functional as F
from torch import nn


# Global dictionary to store feature maps during forward pass
_feature_maps: Dict[str, torch.Tensor] = {}


def get_submodule_by_name(model: nn.Module, layer_name: str) -> nn.Module:
    """
    Retrieve a submodule from a model using its dot-separated name.
    
    Args:
        model: The neural network model
        layer_name: Dot-separated name of the layer (e.g., 'features.denseblock4.denselayer16.conv2')
        
    Returns:
        nn.Module: The requested submodule
        
    Raises:
        AttributeError: If the layer name doesn't exist in the model
    """
    try:
        for attr in layer_name.split("."):
            model = getattr(model, attr)
        return model
    except AttributeError as e:
        raise AttributeError(f"Layer '{layer_name}' not found in model: {e}")


def create_feature_hook(layer_name: str) -> Callable:
    """
    Create a forward hook function that captures and stores layer outputs.
    
    Args:
        layer_name: Name identifier for the layer
        
    Returns:
        Callable: Hook function that can be registered to a layer
    """
    def hook_function(module: nn.Module, input: Tuple, output: torch.Tensor) -> None:
        """Forward hook that stores the output tensor."""
        _feature_maps[layer_name] = output.detach()
    
    return hook_function


def register_feature_hooks(model: nn.Module, layer_names: List[str]) -> List[Any]:
    """
    Register forward hooks on specified layers of the model.
    
    Args:
        model: The neural network model
        layer_names: List of layer names to hook
        
    Returns:
        List of hook handles for cleanup
        
    Raises:
        AttributeError: If any layer name is invalid
    """
    hook_handles = []
    
    for layer_name in layer_names:
        try:
            layer = get_submodule_by_name(model, layer_name)
            hook_handle = layer.register_forward_hook(create_feature_hook(layer_name))
            hook_handles.append(hook_handle)
        except AttributeError:
            # Clean up already registered hooks if one fails
            for handle in hook_handles:
                handle.remove()
            raise
    
    return hook_handles


def extract_gap_features(
    model: nn.Module, 
    input_tensor: torch.Tensor, 
    layer_names: List[str]
) -> Dict[str, torch.Tensor]:
    """
    Extract Global Average Pooled (GAP) features from specified model layers.
    
    This function performs a forward pass through the model while capturing
    intermediate feature maps, then applies global average pooling to create
    compact feature representations.
    
    Args:
        model: The neural network model
        input_tensor: Input tensor of shape [B, C, D, H, W] for 3D models
        layer_names: List of layer names to extract features from
        
    Returns:
        Dict mapping layer names to their GAP feature vectors
        
    Raises:
        RuntimeError: If model forward pass fails
        AttributeError: If layer names are invalid
    """
    # Clear any previous feature maps
    _feature_maps.clear()
    
    # Register hooks on target layers
    hook_handles = register_feature_hooks(model, layer_names)
    
    try:
        # Perform forward pass to capture features
        with torch.no_grad():
            _ = model(input_tensor)
        
        # Apply Global Average Pooling to captured features
        gap_features = {}
        for layer_name, feature_map in _feature_maps.items():
            # Apply 3D adaptive average pooling and flatten
            pooled_feature = F.adaptive_avg_pool3d(feature_map, 1).flatten(1)
            gap_features[layer_name] = pooled_feature
        
        return gap_features
    
    except Exception as e:
        raise RuntimeError(f"Feature extraction failed: {e}")
    
    finally:
        # Always clean up hooks
        for handle in hook_handles:
            handle.remove()
        _feature_maps.clear()


def load_pretrained_model(model_path: str, store_location: str = None) -> nn.Module:
    """
    Load a pre-trained PyTorch model from file.
    
    Args:
        model_path: Name or path of the model file
        store_location: Base directory path (uses STORE_LOCATION env var if None)
        
    Returns:
        nn.Module: The loaded model
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        RuntimeError: If model loading fails
    """
    if store_location is None:
        store_location = os.environ.get("STORE_LOCATION")
        if store_location is None:
            raise EnvironmentError(
                "STORE_LOCATION environment variable not set and no store_location provided"
            )
    
    full_path = os.path.join(store_location, model_path)
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Model file not found: {full_path}")
    
    try:
        model = torch.load(full_path, weights_only=False)
        model.eval()  # Set to evaluation mode
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {full_path}: {e}")


def load_tensor_data(tensor_path: str, store_location: str = None, add_batch_dim: bool = True) -> torch.Tensor:
    """
    Load tensor data from file and optionally add batch dimension.
    
    Args:
        tensor_path: Name or path of the tensor file
        store_location: Base directory path (uses STORE_LOCATION env var if None)
        add_batch_dim: Whether to add a batch dimension to the tensor
        
    Returns:
        torch.Tensor: The loaded tensor
        
    Raises:
        FileNotFoundError: If tensor file doesn't exist
        RuntimeError: If tensor loading fails
    """
    if store_location is None:
        store_location = os.environ.get("STORE_LOCATION")
        if store_location is None:
            raise EnvironmentError(
                "STORE_LOCATION environment variable not set and no store_location provided"
            )
    
    full_path = os.path.join(store_location, tensor_path)
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Tensor file not found: {full_path}")
    
    try:
        tensor = torch.load(full_path)
        
        if add_batch_dim:
            tensor = tensor.unsqueeze(0)
        
        return tensor
    except Exception as e:
        raise RuntimeError(f"Failed to load tensor from {full_path}: {e}")


def print_feature_shapes(feature_dict: Dict[str, torch.Tensor]) -> None:
    """
    Print the shapes of extracted features in a formatted way.
    
    Args:
        feature_dict: Dictionary of layer names to feature tensors
    """
    print("Extracted Feature Shapes:")
    print("-" * 50)
    for layer_name, feature_tensor in feature_dict.items():
        print(f"{layer_name:<40} : {str(feature_tensor.shape)}")


def validate_layer_names(model: nn.Module, layer_names: List[str]) -> List[str]:
    """
    Validate that all specified layer names exist in the model.
    
    Args:
        model: The neural network model
        layer_names: List of layer names to validate
        
    Returns:
        List of valid layer names
        
    Raises:
        AttributeError: If any layer name is invalid
    """
    invalid_layers = []
    
    for layer_name in layer_names:
        try:
            get_submodule_by_name(model, layer_name)
        except AttributeError:
            invalid_layers.append(layer_name)
    
    if invalid_layers:
        raise AttributeError(f"Invalid layer names: {invalid_layers}")
    
    return layer_names


def main() -> None:
    """
    Main function demonstrating feature extraction from a 3D DenseNet model.
    
    This example loads a pre-trained 3D DenseNet121 model and extracts features
    from the last few layers of the final dense block.
    """
    try:
        # Load the pre-trained 3D DenseNet121 model
        print("Loading 3D DenseNet121 model...")
        model_3d = load_pretrained_model("model_3d.pth")
        
        # Load tensor data (femur segmentation example)
        print("Loading tensor data...")
        tensor_3d = load_tensor_data("tensor_3d.pth")
        
        # Define layers for feature extraction (last few layers of DenseNet121)
        target_layers = [
            "features.denseblock4.denselayer16.conv2",  # Final layer
            "features.denseblock4.denselayer15.conv2",  # Third-to-last
            "features.denseblock4.denselayer14.conv2"   # Fifth-to-last
        ]
        
        # Validate layer names
        print("Validating layer names...")
        validate_layer_names(model_3d, target_layers)
        
        # Extract features from specified layers
        print("Extracting features...")
        feature_vectors = extract_gap_features(model_3d, tensor_3d, target_layers)
        
        # Display results
        print_feature_shapes(feature_vectors)
        
        print("\nFeature extraction completed successfully!")
        
    except Exception as e:
        print(f"Error during feature extraction: {e}")
        raise


if __name__ == "__main__":
    main()