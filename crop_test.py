import os, json, cv2, math
import pandas as pd
from tqdm import tqdm

# -------- CONFIG: change these to your paths --------
BASE_DIR     = r"D:\CCTV\shanghaitech"                           # has testing\frames\...
DETS_DIR     = r"D:\CCTV\shanghaitech\outputs\detections" # has testing\...\*.json
OUTPUT_DIR   = r"D:\CCTV\shanghaitech_project\data\crops"         # will save testing crops here
DATASET_SPLIT= "testing"
PADDING_RATIO= 0.15
OUTPUT_SIZE  = 224
# ----------------------------------------------------

def ensure_dir(p): os.makedirs(p, exist_ok=True); return p
def is_image(f): return os.path.splitext(f)[1].lower() in {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}
def clamp(v, lo, hi): return max(lo, min(int(v), hi))

def pad_and_clamp(x1,y1,x2,y2,w,h, pad=PADDING_RATIO):
    bw, bh = x2-x1, y2-y1
    px, py = int(math.floor(bw*pad)), int(math.floor(bh*pad))
    x1p, y1p = clamp(x1-px, 0, w-1), clamp(y1-py, 0, h-1)
    x2p, y2p = clamp(x2+px, 0, w-1), clamp(y2+py, 0, h-1)
    return x1p, y1p, x2p, y2p

def main():
    frames_root = os.path.join(BASE_DIR, DATASET_SPLIT, "frames")
    dets_root   = os.path.join(DETS_DIR, DATASET_SPLIT)
    crops_root  = ensure_dir(os.path.join(OUTPUT_DIR, DATASET_SPLIT))
    index_csv   = os.path.join(crops_root, f"{DATASET_SPLIT}_crops_index.csv")

    if not os.path.isdir(frames_root): raise FileNotFoundError(frames_root)
    if not os.path.isdir(dets_root):   raise FileNotFoundError(dets_root)

    rows = []
    total_imgs, total_crops = 0, 0

    # walk all subfolders under frames_root
    for dirpath, dirnames, filenames in os.walk(frames_root):
        images = [f for f in filenames if is_image(f)]
        if not images:
            continue

        rel = os.path.relpath(dirpath, frames_root)  # relative subpath (could be nested)
        det_dir  = os.path.join(dets_root, rel)
        out_dir  = ensure_dir(os.path.join(crops_root, rel))

        for fname in tqdm(images, desc=f"Cropping {rel}", leave=False):
            total_imgs += 1
            frame_path = os.path.join(dirpath, fname)
            det_path   = os.path.join(det_dir, os.path.splitext(fname)[0] + ".json")

            # if no detection JSON, skip
            if not os.path.isfile(det_path):
                continue

            img = cv2.imread(frame_path)
            if img is None:
                continue
            h, w = img.shape[:2]

            with open(det_path, "r") as f:
                dets = json.load(f)

            obj_id = 0
            for det in dets:
                x1,y1,x2,y2 = det.get("bbox", [0,0,0,0])
                label = det.get("label","")
                conf  = float(det.get("conf",0.0))

                x1p,y1p,x2p,y2p = pad_and_clamp(x1,y1,x2,y2,w,h)
                if x2p<=x1p or y2p<=y1p: continue
                if (x2p-x1p)<5 or (y2p-y1p)<5: continue

                crop = img[y1p:y2p, x1p:x2p]
                crop = cv2.resize(crop, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_LINEAR)

                crop_name = f"{os.path.splitext(fname)[0]}_{obj_id:03d}.jpg"
                crop_path = os.path.join(out_dir, crop_name)
                cv2.imwrite(crop_path, crop)
                obj_id += 1
                total_crops += 1

                rows.append({
                    "split": DATASET_SPLIT,
                    "rel_dir": rel,
                    "frame_file": fname,
                    "frame_path": frame_path,
                    "crop_path": crop_path,
                    "label": label,
                    "conf": conf,
                    "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2,
                    "bbox_x1_pad": x1p, "bbox_y1_pad": y1p, "bbox_x2_pad": x2p, "bbox_y2_pad": y2p,
                    "img_w": w, "img_h": h
                })

    if rows:
        pd.DataFrame(rows).to_csv(index_csv, index=False)
        print(f" Saved index: {index_csv}")
        print(f" Crops root: {crops_root}")
        print(f"Frames processed: {total_imgs} | Crops saved: {total_crops}")
    else:
        print(" No crops created — check that detection JSONs are non-empty and paths are correct.")

if __name__ == "__main__":
    main()
