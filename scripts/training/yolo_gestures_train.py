"""
Fine-tune YOLO11n with a custom dataset containing hand gestures for robot control.
The dataset is sourced from Roboflow and contains 7 classes.
"""
from ultralytics import YOLO
import os

def finetune_gestures():
    print("=== Phase 3.5: Fine-tuning YOLO11n for Hand Gestures ===")
    
    # Path to the data.yaml file
    data_yaml = os.path.abspath("robot-assistant/data/yolo/gestures/Hand-Gesture-Recognition-1/data.yaml")
    
    if not os.path.exists(data_yaml):
        print(f"Error: dataset config not found at {data_yaml}")
        return

    # Load base model
    model = YOLO("yolo11n.pt")

    print("\nStarting training for Hand Gesture Recognition...")
    print("Dataset: Roboflow Hand Gesture Recognition v1")
    print("Epochs: 50, Image Size: 640, Device: CUDA (RTX 3050)")
    
    results = model.train(
        data=data_yaml,
        epochs=50,
        imgsz=640,
        device="cuda",
        batch=16,          # Small batch size for RTX 3050 6GB
        workers=2,
        patience=20,
        save=True,
        project="robot-assistant/models",
        name="yolo11n-gestures",
        exist_ok=True,
        verbose=True,
        lr0=0.01,
        mosaic=1.0,
    )

    print("\n=== Hand Gesture Fine-tuning Complete ===")
    print(f"Best model saved in robot-assistant/models/yolo11n-gestures/weights/best.pt")

if __name__ == "__main__":
    finetune_gestures()
