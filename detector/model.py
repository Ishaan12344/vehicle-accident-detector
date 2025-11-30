import torch
from ultralytics import YOLO

def load_model():
    """
    Load YOLOv8 model on CPU or GPU (if available).
    Returns (model, device_string)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[MODEL] Using device: {device}")
    if device == "cuda":
        print(f"[MODEL] CUDA device: {torch.cuda.get_device_name(0)}")

    # lightweight model, good for real-time
    model = YOLO("yolov8n.pt")  # will auto-download on first run
    model.to(device)
    return model, device
