# 🏠 HomeSense — Live Home Object Finder

HomeSense is a lightweight Python application that uses **YOLOv8** to detect, remember, and locate everyday objects in your home. It scans the environment using a webcam, stores object locations with scene images, and allows users to search for the last known location of an object using **text or voice commands**.

---

## ✨ Features

- 📷 Live webcam object detection using **YOLOv8**
- 🧠 Remembers object locations with scene snapshots
- 🚫 Duplicate detection suppression using **IoU** and center-distance matching
- 💾 Stores object metadata in `data.json`
- 🖼️ Saves captured scene images in the `images/` directory
- 🔍 Search for previously detected objects using text
- 🎤 Voice-based object search
- ⚡ Lightweight and easy-to-run Python project

---

## 📁 Project Structure

```text
HomeSense/
│── images/             # Stored scene images
│── data.json           # Object metadata
│── main.py             # Main application
│── requirements.txt    # Python dependencies
│── README.md
│── yolov8n.pt          # YOLOv8 model (download separately)
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/HomeSense.git
cd HomeSense
```

### 2. Create a Virtual Environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📥 Download the YOLOv8 Model

**Do not commit large model files to GitHub.**

Download the **YOLOv8 Nano** model:

https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

Place the downloaded file (`yolov8n.pt`) in the project root (same folder as `main.py`).

Alternatively, you can upload the model to a **GitHub Release**, **Google Drive**, or another cloud storage service and update this README with the download link.

---

## ▶️ Run the Application

```bash
python main.py
```

The interactive menu allows you to:

- 📷 Start live object scanning
- 👀 Run detection-only mode
- 🔎 Search for stored object locations
- 🎤 Search using voice commands

---

## 💾 Data Storage

The application stores:

- **Object metadata** → `data.json`
- **Captured scene images** → `images/`

Each detected object includes:

- Object name
- Confidence score
- Bounding box coordinates
- Timestamp
- Scene image reference

---

## 🛠️ Technologies Used

- Python
- YOLOv8 (Ultralytics)
- OpenCV
- NumPy
- Speech Recognition
- JSON

---

## 📌 Notes

- Ensure your webcam is connected before running the application.
- The project is optimized for the lightweight **YOLOv8 Nano** model.
- Larger YOLO models can be used for higher detection accuracy but may require more computational resources.

---

## 📄 License

This project is licensed under the MIT License.
