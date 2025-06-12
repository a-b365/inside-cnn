import os
import torch
import pandas as pd
import torch.nn.functional as F

from feature_extraction import extract_features, load_model, load_tensor
from segmentation import create_tensor, watershed_segmentation, load_medical_volume


if __name__=="__main__":

    # Load the 3D densenet121 model
    model_3d = load_model("model_3d.pth")

    tibia_3d = load_tensor("tibia_3d.pth")

    femur_3d = load_tensor("femur_3d.pth")

    background_3d = load_tensor("background_3d.pth")

    layer_names = {
        
        "last": "features.denseblock4.denselayer16.conv2",
        "third-last":"features.denseblock4.denselayer15.conv2",
        "fifth-last":"features.denseblock4.denselayer14.conv2"
    }

    # Region pairs
    region_pairs = [
        ("tibia", "femur"),
        ("tibia", "background"),
        ("femur", "background")
    ]

    # Feature lookup
    feature_lookup = {
        "tibia": extract_features(model_3d, tibia_3d, layer_names.values()),
        "femur": extract_features(model_3d, femur_3d, layer_names.values()),
        "background": extract_features(model_3d, background_3d, layer_names.values())
    }

    # Create a list of rows
    data = []

    # Loop through region pairs and compute cosine similarity
    for r1, r2 in region_pairs:
        row = {}
        row["Image Pair"] = f"{r1} <-> {r2}"
        for pos, layer in layer_names.items():
            f1 = feature_lookup[r1][layer]
            f2 = feature_lookup[r2][layer]
            sim = F.cosine_similarity(f1, f2, dim=1).item()
            row[pos] = sim
        data.append(row)

    df = pd.DataFrame(data)

    df.to_csv(os.getenv("STORE_LOCATION")+"similarity.csv")
