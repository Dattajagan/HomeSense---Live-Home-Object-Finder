import json
import os
import time
from datetime import datetime

import cv2
from ultralytics import YOLO

from speech_engine import listen, speak

MODEL_PATH = "yolov8n.pt"
DATA_PATH = "data.json"
IMAGES_DIR = "images"
MIN_CONFIDENCE = 0.35
STABLE_FRAMES_REQUIRED = 5

SYNONYMS = {
    "phone": "cell phone",
    "cellphone": "cell phone",
    "tv": "tv",
    "television": "tv",
    "refrigerator": "refrigerator",
    "fridge": "refrigerator",
    "mouse": "mouse",
    "laptop": "laptop",
    "pad": "keyboard",
    "mat": "floor mat",
    "book shelf": "book",
}


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
        json.dump(data, f, indent=2)


def format_timestamp(ts=None):
    ts = ts or datetime.now()
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def get_scene_filename(room_name):
    safe_room = room_name.replace(" ", "_").replace("/", "_")
    timestamp = int(time.time())
    return os.path.join(IMAGES_DIR, f"{safe_room}_{timestamp}.jpg")


def normalize_label(label):
    return label.lower().strip()


def normalize_query(query):
    text = query.lower().strip()
    return SYNONYMS.get(text, text)


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 < x1 or y2 < y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    if area1 <= 0 or area2 <= 0:
        return 0.0
    return inter / float(area1 + area2 - inter)


def box_center(box):
    return ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)


def center_distance(box1, box2):
    c1 = box_center(box1)
    c2 = box_center(box2)
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5


def same_object(box1, box2):
    return iou(box1, box2) > 0.45 or center_distance(box1, box2) < 50


def find_existing_entry(label, room_name, box_coords):
    data = load_data()
    normalized_label = normalize_label(label)
    normalized_room = normalize_label(room_name)
    for key, entry in data.items():
        if normalize_label(key).startswith(normalized_label) and normalize_label(entry.get("room", "")) == normalized_room:
            existing_box = entry.get("box", [])
            if existing_box and same_object(existing_box, box_coords):
                return key, entry
    return None, None


def save_single_detection(label, room_name, frame, box_coords, confidence):
    data = load_data()
    ensure_directories()

    scene_path = get_scene_filename(room_name)
    cv2.imwrite(scene_path, frame)

    existing_key, existing_entry = find_existing_entry(label, room_name, box_coords)
    if existing_key:
        if existing_entry:
            existing_entry.update({
                "room": room_name,
                "image": scene_path,
                "box": box_coords,
                "detected_at": format_timestamp(),
                "confidence": round(confidence, 2),
                "source": "auto",
            })
            save_data(data)
            return existing_key

    unique_label = label
    count = 1
    while unique_label in data:
        count += 1
        unique_label = f"{label}_{count}"

    data[unique_label] = {
        "room": room_name,
        "image": scene_path,
        "box": box_coords,
        "detected_at": format_timestamp(),
        "confidence": round(confidence, 2),
        "source": "auto",
    }
    save_data(data)
    return unique_label


def save_detection_results(results, room_name, frame):
    data = load_data()
    ensure_directories()

    scene_path = get_scene_filename(room_name)
    cv2.imwrite(scene_path, frame)

    names = results[0].names if len(results) else {}
    added_labels = []

    for box in results[0].boxes:
        confidence = float(box.conf[0]) if hasattr(box, 'conf') else 1.0
        if confidence < MIN_CONFIDENCE:
            continue

        cls_id = int(box.cls[0])
        label = normalize_label(names.get(cls_id, f"object_{cls_id}"))
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        box_coords = [x1, y1, x2, y2]

        existing_key = None
        for key, entry in data.items():
            if normalize_label(key).startswith(label) and normalize_label(entry.get("room", "")) == normalize_label(room_name):
                if iou(entry.get("box", []), box_coords) > 0.5:
                    existing_key = key
                    break

        if existing_key:
            data[existing_key].update({
                "image": scene_path,
                "box": box_coords,
                "detected_at": format_timestamp(),
                "confidence": round(confidence, 2),
                "source": "auto",
            })
            added_labels.append(existing_key)
            continue

        unique_label = label
        count = 1
        while unique_label in data:
            count += 1
            unique_label = f"{label}_{count}"

        data[unique_label] = {
            "room": room_name,
            "image": scene_path,
            "box": box_coords,
            "detected_at": format_timestamp(),
            "confidence": round(confidence, 2),
            "source": "auto",
        }
        added_labels.append(unique_label)

    if added_labels:
        save_data(data)
    return added_labels


def display_search_result(match, entry):
    print(f"✅ {match} — room: {entry['room']} — last seen: {entry.get('detected_at', 'unknown')} — source: {entry.get('source', 'auto')}")
    image_path = entry.get("image")
    if image_path and os.path.exists(image_path):
        img = cv2.imread(image_path)
        if img is not None:
            box = entry.get("box")
            if box:
                x1, y1, x2, y2 = box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(img, match, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow(match, img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print(f"⚠️ Could not open image: {image_path}")


def find_object(query):
    data = load_data()
    if not data:
        print("No known objects yet. Use scan mode to capture the room.")
        return

    query = normalize_query(query)
    matches = []

    for key, entry in data.items():
        normalized_key = normalize_label(key)
        normalized_room = normalize_label(entry.get("room", ""))
        if query in normalized_key or query in normalized_room:
            matches.append(key)
            continue

        for alias, target in SYNONYMS.items():
            if query == alias and target in normalized_key:
                matches.append(key)
                break

    if not matches:
        print("❌ Object not found in stored memory.")
        return

    print(f"Found {len(matches)} match(es):")
    for match in matches:
        display_search_result(match, data[match])


def show_memory_summary():
    data = load_data()
    if not data:
        print("No object memory available. Please scan a room first.")
        return

    rooms = {}
    for key, entry in data.items():
        rooms.setdefault(entry.get("room", "Unknown"), []).append(key)

    print("Saved object memory:")
    for room, labels in sorted(rooms.items()):
        print(f"- {room}: {', '.join(sorted(labels))}")


def scan_room(room_name):
    ensure_directories()
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Unable to open camera. Make sure a webcam is connected.")
        return

    print("Live scan mode (auto-save ON by default):")
    print("  s = save current detection")
    print("  a = toggle auto-save of new detections")
    print("  q = quit")

    last_print = 0
    auto_save = True
    last_auto_save_time = 0
    auto_save_interval = 5
    candidates = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1
        results = model(frame)
        annotated_frame = results[0].plot()
        labels = []
        detections = []

        for box in results[0].boxes:
            confidence = float(box.conf[0]) if hasattr(box, 'conf') else 1.0
            if confidence < MIN_CONFIDENCE:
                continue
            cls_id = int(box.cls[0])
            label = normalize_label(results[0].names[cls_id])
            bbox = list(map(int, box.xyxy[0].tolist()))
            labels.append(label)
            detections.append({
                "label": label,
                "box": bbox,
                "confidence": confidence,
            })

        now = time.time()
        if now - last_print >= 1:
            if labels:
                prefix = "Auto-save ON:" if auto_save else "Detected:"
                print(f"{prefix} {', '.join(sorted(set(labels))):<60}", end="\r")
            else:
                print("Detecting objects...                              ", end="\r")
            last_print = now

        for candidate in candidates:
            candidate["matched"] = False

        for detection in detections:
            matched = False
            for candidate in candidates:
                if candidate["label"] == detection["label"] and iou(candidate["box"], detection["box"]) > 0.5:
                    candidate["count"] += 1
                    candidate["box"] = [
                        int((candidate["box"][i] + detection["box"][i]) / 2)
                        for i in range(4)
                    ]
                    candidate["confidence"] = max(candidate["confidence"], detection["confidence"])
                    candidate["matched"] = True
                    matched = True
                    break
            if not matched:
                candidates.append({
                    "label": detection["label"],
                    "box": detection["box"],
                    "confidence": detection["confidence"],
                    "count": 1,
                    "matched": True,
                    "saved": False,
                    "missed": 0,
                })

        candidates = [c for c in candidates if c["matched"] or c.setdefault("missed", 0) < 3]
        for candidate in candidates:
            if not candidate["matched"]:
                candidate["missed"] += 1

        if auto_save and detections and frame_count > STABLE_FRAMES_REQUIRED:
            for candidate in candidates:
                if candidate["count"] >= STABLE_FRAMES_REQUIRED and not candidate["saved"]:
                    saved_key = save_single_detection(
                        candidate["label"],
                        room_name,
                        frame,
                        candidate["box"],
                        candidate["confidence"],
                    )
                    if saved_key:
                        print(f"\n✅ Auto-saved: {saved_key}")
                        candidate["saved"] = True
                        candidate["count"] = 0
                    last_auto_save_time = now

        cv2.imshow("Live Scan - Press s to save", annotated_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            added = save_detection_results(results, room_name, frame)
            if added:
                print(f"\n✅ Saved: {', '.join(added)}")
                for candidate in candidates:
                    if candidate["label"] in added:
                        candidate["saved"] = True
            else:
                print("\n⚠️ No objects detected to save.")
        elif key == ord("a"):
            auto_save = not auto_save
            state = "ON" if auto_save else "OFF"
            print(f"\n🔁 Auto-save {state}")
            if auto_save:
                last_auto_save_time = now
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Scan complete.")


def detect_only():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Unable to open camera. Make sure a webcam is connected.")
        return

    print("Live detection only:")
    print("  q = quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        results = model(frame)
        annotated_frame = results[0].plot()
        labels = [results[0].names[int(box.cls[0])] for box in results[0].boxes]

        status = ", ".join(sorted(set(labels))) if labels else "Detecting objects..."
        cv2.putText(annotated_frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Live Detection - Press q to quit", annotated_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def voice_query():
    speak("What object would you like to find?")
    query = listen()
    if query:
        print(f"Searching for: {query}")
        find_object(query)
    else:
        print("No voice query detected.")


def menu():
    while True:
        print("\n=== Home Object Finder ===")
        print("1) Live scan and save object positions")
        print("2) Live object detection only")
        print("3) Search object location by text")
        print("4) Search object location by voice")
        print("5) Show saved object memory")
        print("6) Quit")

        choice = input("Choose an option: ").strip()

        if choice == "1":
            room = input("Room name: ").strip() or "Live"
            scan_room(room)
        elif choice == "2":
            detect_only()
        elif choice == "3":
            query = input("Object name: ").strip()
            if query:
                find_object(query)
        elif choice == "4":
            voice_query()
        elif choice == "5":
            show_memory_summary()
        elif choice == "6":
            print("Goodbye.")
            break
        else:
            print("Invalid choice. Please enter 1-6.")


if __name__ == "__main__":
    menu()
