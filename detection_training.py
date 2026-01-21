# detect_recursive.py
# Usage: edit FRAMES_ROOT and DETS_OUT_ROOT, then run: python detect_recursive.py

import os, json, cv2
from tqdm import tqdm
from ultralytics import YOLO

# ========== EDIT THESE PATHS ==========
FRAMES_ROOT   = r"D:\CCTV\shanghaitech\training\frames"            # input frames root (has many subfolders)
DETS_OUT_ROOT = r"D:\CCTV\shanghaitech\outputs\detections\training" # where JSONs will be written (keeps same subfolders)
# =====================================

# Optional settings (you may tune)
MODEL_WEIGHTS = "yolov8m.pt"   # "yolov8n.pt" (fast), "yolov8l.pt" (more accurate)
CONF_THRES    = 0.15
KEEP_LABELS   = {"person","bicycle","motorbike","car","bus","truck"} 
KEEP_ALL      = False          # True = keep every YOLO class (for debugging)

# --- do not change below unless you know what you do ---
print("Frames root:", FRAMES_ROOT)
print("Detections output:", DETS_OUT_ROOT)
model = YOLO(MODEL_WEIGHTS)

def is_image(name):
    return os.path.splitext(name)[1].lower() in {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}

total = written = 0
if not os.path.isdir(FRAMES_ROOT):
    raise FileNotFoundError("Frames root not found: " + FRAMES_ROOT)
os.makedirs(DETS_OUT_ROOT, exist_ok=True)

for dirpath, _, files in os.walk(FRAMES_ROOT):
    imgs = [f for f in files if is_image(f)]
    if not imgs:
        continue
    rel = os.path.relpath(dirpath, FRAMES_ROOT)
    out_dir = os.path.join(DETS_OUT_ROOT, rel)
    os.makedirs(out_dir, exist_ok=True)

    for fname in tqdm(imgs, desc=f"Detect {rel}", leave=False):
        total += 1
        frame_path = os.path.join(dirpath, fname)
        json_path  = os.path.join(out_dir, os.path.splitext(fname)[0] + ".json")

        img = cv2.imread(frame_path)
        if img is None:
            # write empty JSON to show we visited
            with open(json_path, "w") as f: json.dump([], f)
            continue

        results = model.predict(img, conf=CONF_THRES, verbose=False)
        dets = []
        for r in results:
            if getattr(r, "boxes", None) is None:
                continue
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            for (x1,y1,x2,y2), cls, cf in zip(boxes, classes, confs):
                label = model.names[int(cls)]
                if KEEP_ALL or (label in KEEP_LABELS):
                    dets.append({
                        "label": label,
                        "conf": float(cf),
                        "bbox": [int(x1), int(y1), int(x2), int(y2)]
                    })

        with open(json_path, "w") as f:
            json.dump(dets, f)
        if dets:
            written += 1

print("Done detection.")
print(f"Frames visited: {total} | Frames with >=1 detection: {written}")
print("JSONs saved under:", DETS_OUT_ROOT)
