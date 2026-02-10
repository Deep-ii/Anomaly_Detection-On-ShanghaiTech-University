# crop_from_dets.py
# Usage: edit FRAMES_ROOT, DETS_ROOT, CROPS_OUT_ROOT, then run: python crop_from_dets.py

import os, json, cv2, math
from tqdm import tqdm

# ========== EDIT THESE PATHS ==========
FRAMES_ROOT   = r"D:\CCTV\shanghaitech\training\frames"            # same as used in detection
DETS_ROOT     = r"D:\CCTV\shanghaitech\outputs\detections\training" # where detection jsons were saved
CROPS_OUT_ROOT= r"D:\CCTV\shanghaitech_project\data\crops\training"  # where crops will be written (same subfolders)
# =====================================

# Settings
PADDING_RATIO = 0.15
OUTPUT_SIZE   = 224
MIN_BOX_WH    = 6   # ignore extremely tiny boxes

def ensure_dir(p): os.makedirs(p, exist_ok=True); return p
def is_image(name): return os.path.splitext(name)[1].lower() in {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}
def clamp(v, lo, hi): return max(lo, min(int(v), hi))

def pad_and_clamp(x1,y1,x2,y2,w,h, pad=PADDING_RATIO):
    bw, bh = x2-x1, y2-y1
    px, py = int(math.floor(bw*pad)), int(math.floor(bh*pad))
    x1p = clamp(x1-px, 0, w-1)
    y1p = clamp(y1-py, 0, h-1)
    x2p = clamp(x2+px, 0, w-1)
    y2p = clamp(y2+py, 0, h-1)
    return x1p, y1p, x2p, y2p

# --- main ---
print("Frames root:", FRAMES_ROOT)
print("Detections root:", DETS_ROOT)
print("Crops out root:", CROPS_OUT_ROOT)

if not os.path.isdir(FRAMES_ROOT):
    raise FileNotFoundError("Frames root not found: " + FRAMES_ROOT)
if not os.path.isdir(DETS_ROOT):
    raise FileNotFoundError("Detections root not found: " + DETS_ROOT)

total_frames = 0
frames_with_json = 0
total_crops = 0
rows = []

for dirpath, _, files in os.walk(FRAMES_ROOT):
    imgs = [f for f in files if is_image(f)]
    if not imgs:
        continue
    rel = os.path.relpath(dirpath, FRAMES_ROOT)
    det_dir = os.path.join(DETS_ROOT, rel)
    out_dir = ensure_dir(os.path.join(CROPS_OUT_ROOT, rel))

    for fname in tqdm(sorted(imgs), desc=f"Cropping {rel}", leave=False):
        total_frames += 1
        frame_path = os.path.join(dirpath, fname)
        json_path = os.path.join(det_dir, os.path.splitext(fname)[0] + ".json")

        if not os.path.isfile(json_path):
            # no detection JSON for this frame — skip (or optionally write empty marker)
            continue

        frames_with_json += 1
        img = cv2.imread(frame_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        with open(json_path, "r") as f:
            try: dets = json.load(f)
            except: dets = []

        if not dets:
            continue

        obj_id = 0
        for det in dets:
            x1,y1,x2,y2 = det.get("bbox", [0,0,0,0])
            label = det.get("label","")
            conf  = float(det.get("conf",0.0))
            x1p,y1p,x2p,y2p = pad_and_clamp(int(x1),int(y1),int(x2),int(y2), w, h)
            if x2p<=x1p or y2p<=y1p: continue
            if (x2p-x1p) < MIN_BOX_WH or (y2p-y1p) < MIN_BOX_WH: continue
            crop = img[y1p:y2p, x1p:x2p]
            try:
                crop = cv2.resize(crop, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_LINEAR)
            except:
                continue
            crop_name = f"{os.path.splitext(fname)[0]}_{obj_id:03d}.jpg"
            crop_path = os.path.join(out_dir, crop_name)
            cv2.imwrite(crop_path, crop)
            obj_id += 1
            total_crops += 1

            # optional: record a row (comment out if you don't want CSV)
            rows.append({
                "rel_dir": rel,
                "frame_file": fname,
                "frame_path": frame_path,
                "crop_path": crop_path,
                "label": label,
                "conf": conf,
                "bbox": [int(x1),int(y1),int(x2),int(y2)]
            })

# optional: save index CSV
if rows:
    import pandas as pd
    csv_out = os.path.join(CROPS_OUT_ROOT, "training_crops_index.csv")
    pd.DataFrame(rows).to_csv(csv_out, index=False)
    print("Saved index CSV:", csv_out)

print("Summary:")
print(" total frames scanned:", total_frames)
print(" frames with JSON present:", frames_with_json)
print(" total crops saved:", total_crops)
print(" crops root:", CROPS_OUT_ROOT)
