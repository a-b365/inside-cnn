import os

import torch
from torch import nn
import torchvision.models as models

def inflate_conv2d(conv2d):

    in_channels = conv2d.in_channels
    out_channels = conv2d.out_channels

    k = conv2d.kernel_size[0]
    kernel_size_3d = (k, k, k)

    s = conv2d.stride[0]
    stride_3d = (s, s, s)
    
    if hasattr(conv2d, "padding") and isinstance(conv2d.padding, tuple):
        p = conv2d.padding[0]
        padding_3d = (p, p, p)
    else:
        padding_3d = 0

    conv3d = nn.Conv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size_3d,
        stride=stride_3d,
        padding=padding_3d,
        bias=conv2d.bias
    )

    with torch.no_grad():
        weight_2d = conv2d.weight.data
        weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, k, 1, 1)/k
        conv3d.weight.data.copy_(weight_3d)
        if conv2d.bias is not None:
            conv3d.bias.data.copy_(conv2d.bias.data)
    return conv3d


def inflate_batchnorm2d(bn2d):
    bn3d = nn.BatchNorm3d(
        bn2d.num_features,
        eps=bn2d.eps,
        momentum=bn2d.momentum,
        affine=bn2d.affine,
        track_running_stats=bn2d.track_running_stats
    )

    if bn2d.affine:
        with torch.no_grad():
            bn3d.weight.data.copy_(bn2d.weight.data)
            bn3d.bias.data.copy_(bn2d.bias.data)
    
    if bn2d.track_running_stats:
        bn3d.running_mean.data.copy_(bn2d.running_mean.data)
        bn3d.running_var.copy_(bn2d.running_var.data)

    return bn3d

def inflate_maxpool2d(maxpool2d):

    maxpool3d = nn.MaxPool3d(
        kernel_size=maxpool2d.kernel_size,
        stride=maxpool2d.stride,
        padding=maxpool2d.padding,
        dilation=maxpool2d.dilation,
        ceil_mode=maxpool2d.ceil_mode
    )

    return maxpool3d

def inflate_avgpool2d(avgpool2d):
        
    avgpool3d = nn.AvgPool3d(
        kernel_size=avgpool2d.kernel_size,
        stride=avgpool2d.stride,
        padding=avgpool2d.padding
    )

    return avgpool3d


def convert_densenet121(model_2d):
    for name, module in model_2d.named_children():
        # Recursively go inside children
        convert_densenet121(module)

        # Convert Conv2d → Conv3d
        if isinstance(module, nn.Conv2d):
            new_layer = inflate_conv2d(module)
            setattr(model_2d, name, new_layer)

        # Convert BatchNorm2d → BatchNorm3d
        elif isinstance(module, nn.BatchNorm2d):
            new_layer = inflate_batchnorm2d(module)
            setattr(model_2d, name, new_layer)

        # Convert MaxPool2d → MaxPool3d
        elif isinstance(module, nn.MaxPool2d):
            new_layer = inflate_maxpool2d(module)
            setattr(model_2d, name, new_layer)

        # Convert AvgPool2d → AvgPool3d
        elif isinstance(module, nn.AvgPool2d):
            new_layer = inflate_avgpool2d(module)
            setattr(model_2d, name, new_layer)

    return model_2d


if __name__ == "__main__":

    model_2d = models.densenet121(pretrained=True)
    model_2d.classifier = nn.Identity()
    model_3d = convert_densenet121(model_2d)
    torch.save(model_3d, os.environ["STORE_LOCATION"]+"model.pth")
    
