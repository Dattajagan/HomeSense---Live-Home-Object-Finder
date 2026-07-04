from ultralytics import YOLO
import cv2
import json
import os
import time

DATA_PATH = "data.json"
IMAGES_DIR = "images"


def ensure_directories():
    os.makedirs(IMAGES_DIR, exist_ok=True)


def load_data():
    try:
        with open(DATA_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def save_data(data):
    with open(DATA_PATH, "w") as f:
        json.dump(data, f, indent=4)


def current_timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def scan_and_save_objects(room_name):
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(0)
    ensure_directories()

    data = load_data()

    print("📷 Press 's' to scan and save detected objects")
    print("❌ Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        results = model(frame)
        annotated_frame = results[0].plot()
        cv2.imshow("Object Scanner - Press 's' to save", annotated_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            timestamp = int(time.time())
            scene_path = f"{IMAGES_DIR}/{room_name}_scene_{timestamp}.jpg"
            cv2.imwrite(scene_path, frame)

            names = model.names
            added_labels = []
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                label = names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                unique_label = label
                count = 1
                while unique_label in data:
                    count += 1
                    unique_label = f"{label}_{count}"

                data[unique_label] = {
                    "room": room_name,
                    "image": scene_path,
                    "box": [x1, y1, x2, y2],
                    "detected_at": current_timestamp(),
                }
                added_labels.append(unique_label)

            save_data(data)
            print("✅ Saved:", ", ".join(added_labels))

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("👋 Scan complete.")


if __name__ == "__main__":
    room = input("Enter room name: ").strip()
    scan_and_save_objects(room)
