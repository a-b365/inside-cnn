import torch.nn.functional as F


if __name__=="__main":
    cos_sim = F.cosine_similarity(f1, f2, dim=1)
    print(f"Cosine similarity: {cos_sim.item():.4f}")