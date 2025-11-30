import os
import csv
from datetime import timedelta

import cv2

from .model import load_model

# ---------- SETUP OUTPUT FOLDERS ----------

os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/frames", exist_ok=True)

# ---------- LOAD MODEL ONCE ----------

model, DEVICE = load_model()
VEHICLE_CLASSES = {"car", "motorbike", "bus", "truck"}


# ---------- HELPER FUNCTIONS ----------

def compute_iou(box1, box2):
    """
    Compute IoU between two boxes [x1, y1, x2, y2].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0, (box1[2] - box1[0])) * max(0, (box1[3] - box1[1]))
    area2 = max(0, (box2[2] - box2[0])) * max(0, (box2[3] - box2[1]))

    union_area = area1 + area2 - inter_area + 1e-6
    return inter_area / union_area


def draw_label(frame, text, x1, y1):
    cv2.putText(
        frame,
        text,
        (int(x1), int(y1) - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


# ---------- CORE ACCIDENT LOGIC ----------

def analyze_frames(
    frame_generator,
    base_name,
    fps,
    conf_thres,
    accident_iou_thres,
    area_growth_factor,
    max_frames=None,
):
    """
    Shared logic: run YOLO on frames, detect accidents, save JPG + CSV.

    frame_generator: yields frames (numpy arrays)
    base_name: used for naming outputs
    fps: frames per second (for timestamps)

    Returns: (snapshot_paths, csv_log_path)
    """
    log_csv_path = os.path.join("outputs", f"{base_name}_accident_log.csv")

    prev_boxes = []
    accident_events = []
    snapshot_paths = []
    accident_counter = 0
    frame_idx = 0

    for frame in frame_generator:
        if frame is None:
            break

        frame_idx += 1
        if max_frames is not None and frame_idx > max_frames:
            print(f"[INFO] Reached max_frames={max_frames}, stopping.")
            break

        if frame_idx % 50 == 0:
            print(f"[INFO] Processing frame {frame_idx}...")

        # YOLO inference
        results = model(frame, conf=conf_thres, device=DEVICE, verbose=False)[0]

        curr_boxes = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            if label not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            area = max(0, x2 - x1) * max(0, y2 - y1)
            curr_boxes.append({"coords": [x1, y1, x2, y2], "area": area, "label": label})

        # Heuristic: big overlap + sudden area growth -> possible accident
        accident_in_this_frame = False
        for cb in curr_boxes:
            for pb in prev_boxes:
                iou = compute_iou(cb["coords"], pb["coords"])
                if iou >= accident_iou_thres and pb["area"] > 0:
                    growth = cb["area"] / (pb["area"] + 1e-6)
                    if growth >= area_growth_factor:
                        accident_in_this_frame = True
                        break
            if accident_in_this_frame:
                break

        # Draw boxes
        for cb in curr_boxes:
            x1, y1, x2, y2 = cb["coords"]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            draw_label(frame, cb["label"], x1, y1)

        # Save snapshot on accident
        if accident_in_this_frame:
            accident_counter += 1
            ts = frame_idx / fps if fps > 0 else frame_idx
            time_str = str(timedelta(seconds=int(ts)))

            cv2.putText(
                frame,
                f"ACCIDENT DETECTED! #{accident_counter}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )

            snapshot_path = os.path.join(
                "outputs/frames",
                f"{base_name}_accident_{accident_counter}_frame_{frame_idx}.jpg",
            )
            cv2.imwrite(snapshot_path, frame)
            snapshot_paths.append(snapshot_path)

            accident_events.append(
                {
                    "event_id": accident_counter,
                    "frame": frame_idx,
                    "time_seconds": round(ts, 2),
                    "time_hhmmss": time_str,
                    "snapshot_path": snapshot_path,
                }
            )

        prev_boxes = curr_boxes

    # CSV
    if accident_events:
        with open(log_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "event_id",
                    "frame",
                    "time_seconds",
                    "time_hhmmss",
                    "snapshot_path",
                ],
            )
            writer.writeheader()
            writer.writerows(accident_events)
    else:
        # write just header row when no accidents
        with open(log_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["event_id", "frame", "time_seconds", "time_hhmmss", "snapshot_path"]
            )

    print("[INFO] CSV:", log_csv_path)
    print("[INFO] Snapshots:", len(snapshot_paths))
    return snapshot_paths, log_csv_path


# ---------- MODE 1: UPLOADED VIDEO ----------

def process_uploaded_video(video_file, conf_thres, accident_iou, area_growth):
    """Entry for UI when user uploads a video."""
    if video_file is None:
        return [], None

    video_path = video_file if isinstance(video_file, str) else video_file.name
    if not os.path.exists(video_path):
        return [], None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], None

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"[UPLOAD] {video_path}, fps={fps:.1f}")

    def frame_gen():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame

    snapshots, csv_log = analyze_frames(
        frame_generator=frame_gen(),
        base_name=base_name,
        fps=fps,
        conf_thres=conf_thres,
        accident_iou_thres=accident_iou,
        area_growth_factor=area_growth,
        max_frames=None,
    )

    cap.release()
    return snapshots, csv_log


# ---------- MODE 2: LAPTOP WEBCAM ----------

def process_webcam(duration_sec, conf_thres, accident_iou, area_growth):
    """Entry for UI when user selects laptop webcam."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open laptop webcam (index 0).")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 15.0
    print(f"[WEBCAM] Using FPS={fps:.1f}")

    base_name = "laptop_webcam"
    max_frames = int(fps * duration_sec)

    def frame_gen():
        count = 0
        while count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            count += 1
            yield frame

    snapshots, csv_log = analyze_frames(
        frame_generator=frame_gen(),
        base_name=base_name,
        fps=fps,
        conf_thres=conf_thres,
        accident_iou_thres=accident_iou,
        area_growth_factor=area_growth,
        max_frames=max_frames,
    )

    cap.release()
    return snapshots, csv_log


# ---------- MODE 3: PHONE IP WEBCAM (URL) ----------

def process_phone_ip_cam(ip_url, duration_sec, conf_thres, accident_iou, area_growth):
    """
    Use phone camera via IP Webcam (or similar) app.

    Example URL (from app, on same Wi-Fi):
      http://192.168.1.5:8080/video
    """
    if not ip_url:
        return [], None

    print(f"[PHONE IP CAM] Opening stream: {ip_url}")
    cap = cv2.VideoCapture(ip_url)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open IP camera stream: {ip_url}\n"
            "Check that:\n"
            "  - Phone and laptop are on the same Wi-Fi\n"
            "  - You can open this URL in your browser\n"
            "  - IP Webcam (or similar app) is running."
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 15.0
    print(f"[PHONE IP CAM] Using FPS={fps:.1f}")

    base_name = "phone_ip_cam"
    max_frames = int(fps * duration_sec)

    def frame_gen():
        count = 0
        while count < max_frames:
            ret, frame = cap.read()
            if not ret:
                print("[PHONE IP CAM] No more frames from stream.")
                break
            count += 1
            yield frame

    snapshots, csv_log = analyze_frames(
        frame_generator=frame_gen(),
        base_name=base_name,
        fps=fps,
        conf_thres=conf_thres,
        accident_iou_thres=accident_iou,
        area_growth_factor=area_growth,
        max_frames=max_frames,
    )

    cap.release()
    return snapshots, csv_log
