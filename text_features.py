# embed_text_features.py
import os
import torch
import numpy as np

# ---- EDIT PATHS if needed ----
PROJECT_ROOT = r"D:\CCTV\shanghaitech_project\prompts"
TEXT_DIR = PROJECT_ROOT  # put normal.txt and abnormal.txt here
OUT = os.path.join(r"D:\CCTV\shanghaitech_project\prompts", "finaltake2_text_features.npz")
MODEL_NAME = "ViT-B/32"   # or "ViT-L/14" if you used that originally
# ------------------------------

def read_lines(p):
    with open(p, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    return lines

def main():
    import clip  # from openai/CLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(MODEL_NAME, device=device)

    normal_file = os.path.join(TEXT_DIR, "normal.txt")
    abnormal_file = os.path.join(TEXT_DIR, "abnormal.txt")
    normals = read_lines(normal_file)
    abnormals = read_lines(abnormal_file)

    print("Normals:", len(normals), "Abnormals:", len(abnormals))
    texts = normals + abnormals

    # tokenize and encode in batches
    B = 64
    all_feats = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), B):
            batch = texts[i:i+B]
            tokens = clip.tokenize(batch).to(device)
            feats = model.encode_text(tokens)  # (B, D)
            feats = feats.float()
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats.cpu().numpy())
    text_feats = np.vstack(all_feats).astype(np.float32)

    # save
    np.savez_compressed(OUT, feats=text_feats, n_normal=len(normals), n_abnormal=len(abnormals))
    print("Saved text features to:", OUT)

if __name__ == "__main__":
    main()
