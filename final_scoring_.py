import os
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

# ---------------- CONFIG ----------------
FEATURE_DIR = r"D:\CCTV\shanghaitech_project\data\features"
TEXT_FEATURES = r"D:\CCTV\shanghaitech_project\prompts\finaltake2_text_features.npz"
OUT_DIR = r"D:\CCTV\shanghaitech_project\outputs\scores_take2"

SMOOTH_SIGMA = 21
EPS = 1e-6
# ---------------------------------------


def load_text_features():
    data = np.load(TEXT_FEATURES)
    feats = data["feats"]          # (N_text, D)
    n_normal = int(data["n_normal"])
    n_abnormal = int(data["n_abnormal"])

    normal_feats = feats[:n_normal]
    abnormal_feats = feats[n_normal:n_normal + n_abnormal]

    return normal_feats, abnormal_feats


def compute_object_scores(img_feats, normal_feats, abnormal_feats):
    """
    img_feats: (N_obj, D)
    """
    # cosine similarity (features are already normalized)
    sim_normal = img_feats @ normal_feats.T
    sim_abnormal = img_feats @ abnormal_feats.T

    # paper-style contrastive score
    score = sim_abnormal.max(axis=1) - sim_normal.max(axis=1)

    return score


def process_split(split, norm_mu=None, norm_std=None):
    in_dir = os.path.join(FEATURE_DIR, split)
    out_obj_dir = os.path.join(OUT_DIR, split, "objects")
    out_frm_dir = os.path.join(OUT_DIR, split, "frames")

    os.makedirs(out_obj_dir, exist_ok=True)
    os.makedirs(out_frm_dir, exist_ok=True)

    normal_feats, abnormal_feats = load_text_features()

    all_frame_scores = []

    for fname in tqdm(sorted(os.listdir(in_dir)), desc=f"Scoring {split}"):
        if not fname.endswith(".npz"):
            continue

        data = np.load(os.path.join(in_dir, fname), allow_pickle=True)
        feats = data["feats"]        # (N_obj, D)
        paths = data["paths"]        # crop paths

        # extract frame index from path
        frames = [int(os.path.basename(p).split("_")[0]) for p in paths]

        # object-level score
        obj_scores = compute_object_scores(feats, normal_feats, abnormal_feats)

        obj_df = pd.DataFrame({
            "frame": frames,
            "object_score": obj_scores
        })

        obj_df.to_csv(os.path.join(out_obj_dir, fname.replace(".npz", "_objects.csv")),
                      index=False)

        # frame-level aggregation (MAX)
        frame_scores = obj_df.groupby("frame")["object_score"].max()

        # normalize using training statistics
        if norm_mu is not None:
            frame_scores = (frame_scores - norm_mu) / (norm_std + EPS)

        # temporal smoothing
        smoothed = gaussian_filter1d(frame_scores.values, sigma=SMOOTH_SIGMA)

        frm_df = pd.DataFrame({
            "frame": frame_scores.index.values,
            "raw_score": frame_scores.values,
            "smoothed_score": smoothed
        })

        frm_df.to_csv(os.path.join(out_frm_dir, fname.replace(".npz", "_frames.csv")),
                      index=False)

        all_frame_scores.extend(frame_scores.values.tolist())

    return np.array(all_frame_scores)


def main():
    print(" Scoring TRAINING split (calibration)...")
    train_scores = process_split("training")

    mu = train_scores.mean()
    std = train_scores.std()

    print(f"Calibration stats → mean: {mu:.4f}, std: {std:.4f}")

    print(" Scoring TESTING split (paper-style)...")
    process_split("testing", norm_mu=mu, norm_std=std)

    print(" Scoring complete")


if __name__ == "__main__":
    main()
