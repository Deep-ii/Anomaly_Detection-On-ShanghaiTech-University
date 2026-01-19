# Anomaly_Detection-On-ShanghaiTech-University
This repository implements a zero-shot, vision–language-based anomaly detection system for surveillance videos. The approach leverages CLIP image–text embeddings, object-level analysis, and contrastive scoring to detect abnormal events (e.g., fights, vehicles, panic) in a university campus environment without training on abnormal samples.
📌 Project Title

Text-Guided Video Anomaly Detection using CLIP on ShanghaiTech Campus Dataset

🔍 Overview

This project presents a zero-shot video anomaly detection framework based on vision–language alignment. Instead of training a supervised anomaly classifier, the system compares object-level visual features extracted from surveillance footage with textual descriptions of normal and abnormal behaviors using the CLIP model.
The method is inspired by recent research on text-guided anomaly detection and is evaluated on the ShanghaiTech Campus Dataset, a standard benchmark for video anomaly detection.

🎯 Objectives
Detect anomalous events in campus surveillance videos
Avoid training on abnormal samples
Leverage natural language descriptions for anomaly semantics
Provide frame-level anomaly scores and visualization
Follow research-paper-aligned scoring logic

📂 Dataset
ShanghaiTech Campus Dataset
Real-world CCTV surveillance videos
Training set contains only normal events
Testing set contains both normal and abnormal events
Frame-level ground truth masks provided

📎 Dataset link:
👉 https://svip-lab.github.io/dataset/campus_dataset.html
⚠️ Due to size constraints, the dataset is not included in this repository.

🏗️ System Pipeline
Video 
→ Frames → Object Detection (YOLO)
→ Object Cropping
→ CLIP Image Embeddings
→ CLIP Text Embeddings (Normal / Abnormal)
→ Contrastive Scoring
→ Frame Aggregation + Z-score
→ Temporal Smoothing
→ Visualization & Evaluation

🧠 Methodology
1. Object-Level Processing
Objects (persons, bicycles, vehicles) are detected using YOLO and cropped from each frame. This allows the system to focus on localized behaviors where anomalies typically occur.

2. Feature Extraction
Image features are extracted using the frozen CLIP image encoder.
Text features are extracted from carefully engineered prompts describing normal and abnormal campus behaviors.

3. Anomaly Scoring
For each object embedding:
score = max(sim(image, abnormal_text)) − max(sim(image, normal_text))
Frame-level anomaly scores are obtained by taking the maximum object score per frame.

4. Calibration & Smoothing
Z-score normalization is applied using statistics from training videos.
Gaussian temporal smoothing improves stability.

📊 Evaluation
Primary metric: Frame-level ROC-AUC
Secondary metric: Frame-level accuracy at a chosen Z-score threshold

🖥️ Visualization
The system provides:
Live video playback with anomaly labels
A real-time anomaly score graph
Clear visualization of abnormal events

⚠️ Limitations
CLIP is not trained specifically for surveillance anomalies
Performance depends heavily on prompt quality
No explicit temporal learning (frame-based analysis)

🔮 Future Work
Prototype-based fine-tuning of text embeddings
Integration of motion features (optical flow)
Temporal modeling using transformers
Adaptive threshold selection

📌 Key Takeaway
This project demonstrates that vision–language models can be effectively used for zero-shot anomaly detection in real-world surveillance scenarios when combined with object-level analysis and carefully engineered text prompts.

👤 Author
Deep Isalaniya
deep2479mission@gmail.com
