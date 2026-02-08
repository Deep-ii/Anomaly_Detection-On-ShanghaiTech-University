import cv2
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque

# ---------------- CONFIG ----------------
FRAME_DIR = Path(r"D:\CCTV\shanghaitech\testing\frames\05_0019")
SCORE_CSV = Path(r"D:\CCTV\shanghaitech_project\outputs\scores_take2\testing\frames\testing_05_0019_frames.csv")

FPS = 15
ANOMALY_Z_TH = 1.0
WINDOW = 200
# --------------------------------------

# Load scores
df = pd.read_csv(SCORE_CSV)
frame_to_score = dict(zip(df["frame"], df["smoothed_score"]))

# Load frames
frame_paths = sorted(FRAME_DIR.glob("*.jpg"))

# -------- Graph setup --------
plt.ion()
fig, ax = plt.subplots(figsize=(6, 4))

x_buf = deque(maxlen=WINDOW)
y_buf = deque(maxlen=WINDOW)

(line,) = ax.plot([], [], lw=2)
ax.axhline(ANOMALY_Z_TH, color="r", linestyle="--")

ax.set_ylim(-2, 6)
ax.set_xlabel("Frame")
ax.set_ylabel("Z-score")
ax.set_title("Live Anomaly Score")

# -------- Main loop --------
for i, frame_path in enumerate(frame_paths):
    frame = cv2.imread(str(frame_path))
    if frame is None:
        continue

    score = frame_to_score.get(i, 0.0)

    # Update graph
    x_buf.append(i)
    y_buf.append(score)
    line.set_data(x_buf, y_buf)
    ax.set_xlim(max(0, i - WINDOW), i + 1)

    plt.pause(0.001)   # REQUIRED for live graph

    # Anomaly logic
    if score > ANOMALY_Z_TH:
        label = "ANOMALY"
        color = (0, 0, 255)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), color, 6)
    else:
        label = "NORMAL"
        color = (0, 255, 0)

    cv2.putText(frame, label, (40, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)

    cv2.putText(frame, f"Score: {score:.3f}", (40, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    cv2.imshow("Live CCTV (Frames)", frame)

    if cv2.waitKey(int(1000 / FPS)) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
plt.ioff()
plt.show()
