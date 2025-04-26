# 🚀 Real-Time Object Detection & Tracking System (YOLOv8 + BoT-SORT)

This project is a real-time object detection and tracking system that identifies **newly appearing** and **missing objects** in video streams. Built using YOLOv8 for detection and BoT-SORT for robust multi-object tracking, it also logs events and outputs annotated video with visual overlays.

---

## 📌 Features

- 🎯 YOLOv8 for accurate, high-speed object detection
- 🧠 BoT-SORT for efficient multi-object tracking
- 🧾 Logs new and missing objects to JSON files
- 📦 Supports both webcam and video file input
- 📽️ Real-time annotated output with FPS counter
- 🐳 Docker support for quick deployment

---

## 📸 Sample Output

Annotated frames and a sample output video are available in the repository.

---

## ⚙️ Installation & Usage

You can run the system in **two ways**:

---

### 🔧 Method 1: Run via Python (Manual Setup)

#### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/real-time-object-tracker.git
cd real-time-object-tracker
```

#### Step 2: Create a Virtual Environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

#### Step 4: Run the Script

- Run with Webcam + Logs + Output Video

```bash
python main.py --source 0 --output output.mp4 --log-dir "my_object_logs"
```

- Run with an Input Video File

```bash
python main.py --source "sample.mp4" --output output/output.mp4
```

---

### 🐳 Method 2: Run via Docker

#### Step 1: Build Docker Image

```bash
docker build -t object-tracker .
```

#### Step 2: Run the Container

```bash
docker run --rm -v ${PWD}:/app -it object-tracker \
  python main.py --source sample.mp4 --output output/output.mp4 --log-dir object_logs
```

---



## 💻 Tested On

| Component      | Specification             |
|----------------|---------------------------|
| **CPU**        | AMD Ryzen 5 3500U         |
| **GPU**        | Integrated Radeon Vega 8  |
| **RAM**        | 8 GB                      |
| **OS**         | Windows/Linux             |
| **FPS**        | ~4.8 FPS (YOLOv8n)

---


