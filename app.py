import streamlit as st
import pandas as pd
import cv2
import time
import requests
import json
from pathlib import Path

# ================= CONFIG =================
st.set_page_config(
    page_title="AI CCTV Anomaly Detection",
    layout="wide"
)

TELEGRAM_TOKEN = "7899528274:AAGvIvalLSB-aWOLqUg6JzqkBNsPUHW68NA"
SUBSCRIBER_FILE = "subscribers.json"

ALERT_EVERY_N_FRAMES = 15   # send alert every N abnormal frames

# ================= TELEGRAM FUNCTION =================
def send_telegram_alert(frame_path, score, frame_idx, abnormal_count):

    # Load subscribers
    try:
        with open(SUBSCRIBER_FILE, "r") as f:
            subscribers = json.load(f).get("subscribers", [])
    except:
        subscribers = []

    if not subscribers:
        print(" No subscribers registered")
        return

    caption = (
        "🚨 CCTV ANOMALY ALERT 🚨\n\n"
        f"Frame Index: {frame_idx}\n"
        f"Anomaly Score: {score:.3f}\n"
        f"Continuous Abnormal Frames: {abnormal_count}\n"
        "Status: Ongoing abnormal activity"
    )

    for chat_id in subscribers:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"

            with open(frame_path, "rb") as img:
                requests.post(
                    url,
                    data={"chat_id": chat_id, "caption": caption},
                    files={"photo": img}
                )

            print(f" Alert sent to {chat_id}")

        except Exception as e:
            print(f" Telegram error for {chat_id}: {e}")

# ================= SIDEBAR =================
st.sidebar.title("⚙ Configuration")

frames_dir = Path(st.sidebar.text_input(
    "Frames folder",
    r"D:\CCTV\shanghaitech\testing\frames\05_0019"
))

score_csv = Path(st.sidebar.text_input(
    "Score CSV",
    r"D:\CCTV\shanghaitech_project\outputs\scores_take2\testing\frames\testing_05_0019_frames.csv"
))

threshold = st.sidebar.slider("🚨 Anomaly Threshold", 0.0, 3.0, 1.0)
fps = st.sidebar.slider("🎥 Playback FPS", 1, 24, 24)
start = st.sidebar.button("▶ Start Monitoring")

# ================= VALIDATION =================
if not frames_dir.exists() or not score_csv.exists():
    st.warning("⚠ Please provide valid paths")
    st.stop()

# ================= LOAD DATA =================
df = pd.read_csv(score_csv)
scores = df["smoothed_score"].values
frames = sorted(frames_dir.glob("*.jpg"))

# ================= UI LAYOUT =================
left, right = st.columns([1.4, 1])

video_box = left.empty()
status_box = left.empty()

chart_df = pd.DataFrame({"Score": []})
chart = right.line_chart(chart_df)

# ================= MAIN LOOP =================
if start:

    abnormal_count = 0

    for i in range(min(len(frames), len(scores))):

        # -------- FRAME --------
        img = cv2.imread(str(frames[i]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        video_box.image(img, use_container_width=True)

        score = float(scores[i])

        # -------- STATUS --------
        if score >= threshold:
            abnormal_count += 1
            status_box.error(
                f" ANOMALY DETECTED\n\n"
                f"Score: {score:.3f}\n"
                f"Abnormal Frames: {abnormal_count}",
                icon="🚨"
            )

            #  SEND ALERT EVERY N ABNORMAL FRAMES
            if abnormal_count % ALERT_EVERY_N_FRAMES == 0:
                send_telegram_alert(
                    frame_path=str(frames[i]),
                    score=score,
                    frame_idx=i,
                    abnormal_count=abnormal_count
                )

        else:
            abnormal_count = 0
            status_box.success(
                f" NORMAL ACTIVITY\n\nScore: {score:.3f}",
                icon="✅"
            )

        # -------- GRAPH UPDATE --------
        chart.add_rows(pd.DataFrame({"Score": [score]}))

        time.sleep(1 / fps)
