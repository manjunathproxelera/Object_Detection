# Object Detection Application (DeepX + YOLOv5)

## 📌 Overview

This project performs object detection using a YOLOv5 model compiled with DeepX (`.dxnn`). It supports both:

* Image inference
* Video / Camera inference (Raspberry Pi supported)

---

## 📁 Project Structure

```
Object_detection/
│
├── main.py                 # Main script
├── config.py               # Preprocess, postprocess, drawing functions
├── coco_classes.json       # Class labels
├── src/output/             # Output images/videos
└── models/                 # Compiled .dxnn models
```

---

## 🚀 Running the Application

### 1️⃣ Activate Environment

```bash
source object-detect-env/bin/activate
```

---

### 2️⃣ Run Script

```bash
python3 demo.py
```

You will see:

```
1. Run Object Detection on image
2. Run Object Detection on video/camera
3. Exit
```

---

## 🖼️ Image Inference

### Steps:

1. Select option `1`
2. The model runs on the input image
3. Output image is saved

### Output:

* Saved at: `src/output/5-out.jpg`

---

## 🎥 Video / Camera Inference

### Steps:

1. Select option `2`
2. Camera/video stream starts
3. Detection runs frame-by-frame
4. Output video is saved

### Output:

* Saved at: `src/output/<timestamp>.mp4`

---

## ⌨️ Controls

| Key   | Action |
| ----- | ------ |
| `q`   | Quit   |
| `Esc` | Exit   |

---







