"""
Model Inflation Pipeline

This module provides utilities to convert 2D PyTorch models to their 3D counterparts
by "inflating" 2D layers (Conv2d, BatchNorm2d, etc.) to 3D equivalents (Conv3d, 
BatchNorm3d, etc.). This is particularly useful for medical image analysis where
volumetric data processing is required.

The inflation process preserves the pretrained weights from 2D models while adapting
them for 3D input data.

Author: Amir Bhattarai
Date: June 12, 2025
Version: 1.0

"""

import os
from typing import Optional

import torch
from torch import nn
import torchvision.models as models


def inflate_conv2d(conv2d: nn.Conv2d) -> nn.Conv3d:
    """
    Convert a 2D convolutional layer to a 3D convolutional layer.
    
    Args:
        conv2d: The 2D convolutional layer to be converted
        
    Returns:
        nn.Conv3d: The equivalent 3D convolutional layer with inflated weights
    """
    in_channels = conv2d.in_channels
    out_channels = conv2d.out_channels
    
    # Extract kernel size and create 3D equivalent
    k = conv2d.kernel_size[0]
    kernel_size_3d = (k, k, k)
    
    # Extract stride and create 3D equivalent
    s = conv2d.stride[0]
    stride_3d = (s, s, s)
    
    # Handle padding - check if it's a tuple or single value
    if hasattr(conv2d, "padding") and isinstance(conv2d.padding, tuple):
        p = conv2d.padding[0]
        padding_3d = (p, p, p)

    else:
        padding_3d = 0
    
    # Create the 3D convolutional layer
    conv3d = nn.Conv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size_3d,
        stride=stride_3d,
        padding=padding_3d,
        bias=conv2d.bias is not None
    )
    
    # Copy weights from 2D to 3D
    with torch.no_grad():
        weight_2d = conv2d.weight.data
        # Inflate weights by repeating along the depth dimension and averaging
        weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, k, 1, 1) / k
        conv3d.weight.data.copy_(weight_3d)
        
        # Copy bias if it exists
        if conv2d.bias is not None:
            conv3d.bias.data.copy_(conv2d.bias.data)
    
    return conv3d


def inflate_batchnorm2d(bn2d: nn.BatchNorm2d) -> nn.BatchNorm3d:
    """
    Convert a 2D batch normalization layer to a 3D batch normalization layer.
    
    Args:
        bn2d: The 2D batch normalization layer to be converted
        
    Returns:
        nn.BatchNorm3d: The equivalent 3D batch normalization layer
    """
    bn3d = nn.BatchNorm3d(
        bn2d.num_features,
        eps=bn2d.eps,
        momentum=bn2d.momentum,
        affine=bn2d.affine,
        track_running_stats=bn2d.track_running_stats
    )
    
    # Copy affine parameters if they exist
    if bn2d.affine:
        with torch.no_grad():
            bn3d.weight.data.copy_(bn2d.weight.data)
            bn3d.bias.data.copy_(bn2d.bias.data)
    
    # Copy running statistics if tracking is enabled
    if bn2d.track_running_stats:
        bn3d.running_mean.data.copy_(bn2d.running_mean.data)
        bn3d.running_var.data.copy_(bn2d.running_var.data)
    
    return bn3d


def inflate_maxpool2d(maxpool2d: nn.MaxPool2d) -> nn.MaxPool3d:
    """
    Convert a 2D max pooling layer to a 3D max pooling layer.
    
    Args:
        maxpool2d: The 2D max pooling layer to be converted
        
    Returns:
        nn.MaxPool3d: The equivalent 3D max pooling layer
    """
    maxpool3d = nn.MaxPool3d(
        kernel_size=maxpool2d.kernel_size,
        stride=maxpool2d.stride,
        padding=maxpool2d.padding,
        dilation=maxpool2d.dilation,
        ceil_mode=maxpool2d.ceil_mode
    )
    
    return maxpool3d


def inflate_avgpool2d(avgpool2d: nn.AvgPool2d) -> nn.AvgPool3d:
    """
    Convert a 2D average pooling layer to a 3D average pooling layer.
    
    Args:
        avgpool2d: The 2D average pooling layer to be converted
        
    Returns:
        nn.AvgPool3d: The equivalent 3D average pooling layer
    """
    avgpool3d = nn.AvgPool3d(
        kernel_size=avgpool2d.kernel_size,
        stride=avgpool2d.stride,
        padding=avgpool2d.padding
    )
    
    return avgpool3d


def convert_densenet121(model_2d: nn.Module) -> nn.Module:
    """
    Recursively convert a 2D DenseNet121 model to its 3D equivalent.
    
    This function traverses the model architecture and replaces all 2D layers
    with their 3D counterparts while preserving the pretrained weights.
    
    Args:
        model_2d: The 2D DenseNet121 model to be converted
        
    Returns:
        nn.Module: The converted 3D model
    """
    for name, module in model_2d.named_children():
        # Recursively process child modules
        convert_densenet121(module)
        
        # Convert different layer types
        if isinstance(module, nn.Conv2d):
            new_layer = inflate_conv2d(module)
            setattr(model_2d, name, new_layer)
            
        elif isinstance(module, nn.BatchNorm2d):
            new_layer = inflate_batchnorm2d(module)
            setattr(model_2d, name, new_layer)
            
        elif isinstance(module, nn.MaxPool2d):
            new_layer = inflate_maxpool2d(module)
            setattr(model_2d, name, new_layer)
            
        elif isinstance(module, nn.AvgPool2d):
            new_layer = inflate_avgpool2d(module)
            setattr(model_2d, name, new_layer)
    
    return model_2d


def save_model(model: nn.Module, save_path: str) -> None:
    """
    Save the converted 3D model to the specified path.
    
    Args:
        model: The model to be saved
        save_path: Full path where the model should be saved
        
    Raises:
        OSError: If the directory doesn't exist or there are permission issues
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model, save_path)
    print(f"Model saved successfully to: {save_path}")


def main() -> None:
    """
    Main function to load, convert, and save the DenseNet121 model.
    
    Loads a pretrained 2D DenseNet121, removes the classifier, converts it to 3D,
    and saves the result to the location specified by the STORE_LOCATION environment variable.
    """
    # Load pretrained 2D DenseNet121
    model_2d = models.densenet121(pretrained=True)
    
    # Remove the classifier layer (replace with identity)
    model_2d.classifier = nn.Identity()
    
    # Convert to 3D model
    print("Converting 2D DenseNet121 to 3D...")
    model_3d = convert_densenet121(model_2d)
    
    # Get save location from environment variable
    store_location = os.environ.get("STORE_LOCATION")
    if store_location is None:
        raise EnvironmentError(
            "STORE_LOCATION environment variable is not set. "
            "Please set it to specify where to save the model."
        )
    
    # Create full save path
    save_path = os.path.join(store_location, "densenet121_3d.pth")
    
    # Save the converted model
    save_model(model_3d, save_path)
    print("Model conversion completed successfully!")


if __name__ == "__main__":
    main()
