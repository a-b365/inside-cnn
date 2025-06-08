import os
import torch
import torch.nn.functional as F

feature_maps = {}

def get_hook(name):
    def hook(module, input, output):
        feature_maps[name] = output.detach()
    return hook

if __name__ == "__main__":
    
    model_3d = torch.load(os.environ["STORE_LOCATION"]+"model.pth")
    
    tensor_3d = torch.load(os.environ["STORE_LOCATION"]+"tensor_3d.pth")
    tensor_3d = tensor_3d.unsqueeze(0)

    model_3d.features.denseblock4.denselayer16.conv2.register_forward_hook(get_hook("last-convolution-layer"))
    model_3d.features.denseblock4.denselayer15.conv2.register_forward_hook(get_hook("third-last-convolution-layer"))
    model_3d.features.denseblock4.denselayer14.conv2.register_forward_hook(get_hook("fifth-last-convolution-layer"))
    
    model_3d.eval()
    with torch.no_grad():
        _ = model_3d(tensor_3d)

    gap = F.adaptive_avg_pool3d(feature_maps["last-convolution-layer"], output_size=1)
    gap = gap.view(gap.size(0), -1)
    gap.shape