import os
import torch
import torch.nn.functional as F

from feature_extraction import extract_features, load_model, load_tensor
from segmentation import create_tensor, watershed_segmentation, load_medical_volume

if __name__=="__main":

    # Load the 3D densenet121 model
    model_3d = load_model("model_3d.pth")

    mask_path = os.environ.get("MASK_DATA_PATH")

    mask_data, _ = load_medical_volume(mask_path)

    labels = watershed_segmentation(mask_data)

    tibia_3d = create_tensor(labels, "tibia")

    femur_3d = create_tensor(labels, "femur")

    background_3d = create_tensor(labels, "background_3d")

    layer_names = [
        
        "features.denseblock4.denselayer16.conv2",
        "features.denseblock4.denselayer15.conv2",
        "features.denseblock4.denselayer14.conv2"
    ]

    features = {
        
        "tibia": extract_features(model_3d, tibia_3d, layer_names),

        # "femur": extract_features(model_3d, femur_3d, layer_names),
        
        # "background" : extract_features(model_3d, background_3d, layer_names)

    }

    for name, ndim in features.items():
        for _, feature in ndim.items():
            print(feature)

    cos_sim = F.cosine_similarity(f1, f2, dim=1)
    print(f"Cosine similarity: {cos_sim.item():.4f}")