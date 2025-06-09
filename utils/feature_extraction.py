import os
import torch
import torch.nn.functional as F
import mayavi.mlab as mlab

from segmentation import create_tensor, watershed_segmentation, load_medical_volume

feature_maps = {}

def get_submodule(model, layer_name):
    """Fetch a submodule from a model using its dot-separated name."""
    for attr in layer_name.split("."):
        model = getattr(model, attr)
    return model

def get_hook(layer_name: str):
    """Create a forward hook that saves output features."""
    def hook(module, input, output):
        feature_maps[layer_name] = output.detach()
    return hook

def register_hooks(model, layer_names):
    """Register hooks on the selected layers and return the hook handles."""
    hooks = []
    for layer_name in layer_names:
        layer = get_submodule(model, layer_name)
        hooks.append(layer.register_forward_hook(get_hook(layer_name)))
    return hooks

def extract_features(model, tensor, layer_names):
    """
    Extract GAP features from specified layers of the model.

    Args:
        model (torch.nn.Module): The neural network model.
        x (torch.Tensor): Input tensor of shape [B, C, D, H, W].
        layer_names (list[str]): Names of layers to extract features from.

    Returns:
        dict[str, torch.Tensor]: Dictionary of GAP feature vectors.
    """
    feature_maps.clear()  # Clear previous runs
    hooks = register_hooks(model, layer_names)

    with torch.no_grad():
        _ = model(tensor)

    for hook in hooks:
        hook.remove()

    pooled = {
        name: F.adaptive_avg_pool3d(feature, 1).flatten(1)
        for name, feature in feature_maps.items()
    }
    return pooled  

def load_model(name:str):
    model_3d = torch.load(os.environ["STORE_LOCATION"]+name, weights_only=False)
    return model_3d

def load_tensor(name:str):
    tensor_3d = torch.load(os.environ["STORE_LOCATION"]+name)
    tensor_3d = tensor_3d.unsqueeze(0)     # Prepare input tensor by adding batch dimension
    return tensor_3d


if __name__ == "__main__":

    # Load the 3D densenet121 model
    model_3d = load_model("model_3d.pth")
    
    # Load the tensor with segmentation results
    tensor_3d = load_tensor("femur_3d.pth")

    layer_names = [
        
        "features.denseblock4.denselayer16.conv2",
        "features.denseblock4.denselayer15.conv2",
        "features.denseblock4.denselayer14.conv2"
    ]

    feature_vectors = extract_features(model_3d, tensor_3d, layer_names)

    for name, vec in feature_vectors.items():
        print(f"{name}: {vec.shape}")
