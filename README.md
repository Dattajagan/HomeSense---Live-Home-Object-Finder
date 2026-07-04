HomeSense — Live Home Object Finder
HomeSense is a small Python tool that uses YOLOv8 to detect and remember object locations around your home. It can run live from a webcam, save detected objects with screenshots, and let you query where an object was last seen (text or voice).

Features
Live camera scanning with stable auto-save heuristics
Duplicate suppression using IoU / center-distance
Saves metadata in data.json and scene images in images
Text and voice search of stored object locations
Minimal dependencies and easy local use
Quick start
Clone the repo:

Create a Python environment and install dependencies:

Provide the YOLOv8 model

Do NOT commit large model files to GitHub.
Download yolov8n.pt (YOLOv8 nano) and place it next to main.py:
Alternatively host the model in a GitHub Release or cloud storage and add a download link here.
Run:

How data is stored
data.json — stores detections and metadata (label, room, box coords, image path, timestamp, confidence).
images/ — scene images saved by the app.
images_archive/ — archived original images (if any). Large images should remain local and not be pushed to GitHub.
