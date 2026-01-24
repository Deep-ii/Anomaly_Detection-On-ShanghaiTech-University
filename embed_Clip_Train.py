import os
import sys
import torch
import clip
import numpy as np
from PIL import Image
from tqdm import tqdm

# -------- CONFIG: set these to your paths --------
CROPS_ROOT = r"D:\CCTV\shanghaitech\data\crops"      # contains training\ and testing\
PROMPTS_DIR = r"D:\CCTV\shanghaitech_project\prompts"
FEATURES_ROOT = r"D:\CCTV\shanghaitech_project\data\features"
SPLIT = "training"   # <-- set to "training" now
MODEL_NAME = "ViT-B/32"
BATCH = 128
IMG_EXTS = {".jpg", ".jpeg", ".png"}   # supported crop extensions
# -------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(FEATURES_ROOT, exist_ok=True)

def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]

def safe_relpath(path, start):
    # Windows-safe relpath
    return os.path.relpath(os.path.abspath(path), os.path.abspath(start))

def embed_texts_once():
    """Create text_features.npz if missing."""
    out_npz = os.path.join(FEATURES_ROOT, "text_features.npz")
    if os.path.isfile(out_npz):
        print("text_features.npz already exists — skipping.")
        return
    print("Encoding text prompts...")
    model, _ = clip.load(MODEL_NAME, device=device)
    normals = read_lines(os.path.join(PROMPTS_DIR, "normal.txt"))
    abnormals = read_lines(os.path.join(PROMPTS_DIR, "abnormal.txt"))
    texts = normals + abnormals
    with torch.no_grad():
        toks = clip.tokenize(texts, truncate=True).to(device)
        feats = []
        for i in range(0, len(texts), BATCH):
            f = model.encode_text(toks[i:i+BATCH])
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.float().cpu().numpy())
        feats = np.concatenate(feats, axis=0)
    np.savez(out_npz, feats=feats, n_normal=len(normals),
             n_abnormal=len(abnormals), model=MODEL_NAME)
    print("Saved:", out_npz)

def iter_image_folders(split_root):
    """Yield (rel_subdir, [image_paths...]) for every subdir that has images."""
    for dirpath, _, files in os.walk(split_root):
        imgs = [os.path.join(dirpath, f) for f in files
                if os.path.splitext(f)[1].lower() in IMG_EXTS]
        if imgs:
            rel = safe_relpath(dirpath, split_root)  # may be "."
            yield rel, sorted(imgs)

def embed_split():
    # paths
    split_crops = os.path.join(CROPS_ROOT, SPLIT)
    split_feats_dir = os.path.join(FEATURES_ROOT, SPLIT)
    os.makedirs(split_feats_dir, exist_ok=True)

    if not os.path.isdir(split_crops):
        raise FileNotFoundError(f"Crop folder not found: {split_crops}")

    # load CLIP once
    model, preprocess = clip.load(MODEL_NAME, device=device)
    total_imgs, written_files = 0, 0

    folders = list(iter_image_folders(split_crops))
    if not folders:
        print(f" No images found under: {split_crops}")
        print("   Check that your crops exist and extensions are .jpg/.jpeg/.png")
        return

    print(f"Found {len(folders)} crop folders in {SPLIT}.")
    for rel, imgs in tqdm(folders, desc=f"Embedding {SPLIT}"):
        total_imgs += len(imgs)

        # one output npz per folder (flatten path with underscores)
        out_name = (rel.replace("\\", "_").replace("/", "_") or "root") + ".npz"
        out_npz = os.path.join(split_feats_dir, out_name)
        if os.path.isfile(out_npz):
            # already done; skip to be resume-safe
            continue

        feats, paths = [], []
        with torch.no_grad():
            for i in range(0, len(imgs), BATCH):
                batch_paths = imgs[i:i+BATCH]
                batch_imgs = [preprocess(Image.open(p).convert("RGB")) for p in batch_paths]
                batch_tensor = torch.stack(batch_imgs).to(device)
                f = model.encode_image(batch_tensor)
                f = f / f.norm(dim=-1, keepdim=True)
                feats.append(f.float().cpu().numpy())
                paths.extend(batch_paths)
        feats = np.concatenate(feats, axis=0) if feats else np.zeros((0,512), dtype=np.float32)
        np.savez(out_npz, feats=feats, paths=np.array(paths))
        written_files += 1

        # helpful print for debugging
        print(f"[{SPLIT}] wrote {out_npz} | imgs: {len(imgs)}")

    print(f" Done {SPLIT}: folders={len(folders)}, total_imgs={total_imgs}, files_written={written_files}")
    print(f"Outputs in: {split_feats_dir}")

if __name__ == "__main__":
    embed_texts_once()
    embed_split()
