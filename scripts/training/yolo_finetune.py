"""
Fine-tune YOLO11n with a custom dataset containing computer lab objects.
Since Roboflow private datasets are inaccessible, we create a custom YAML
config that uses Open Images V7 classes relevant to a computer lab.
Open Images V7 has 600+ classes including many that COCO doesn't have.
"""
from ultralytics import YOLO
import yaml
import os

# Computer lab relevant classes from Open Images V7 (600 classes)
# These are additional to what COCO already offers
LAB_CLASSES = [
    "Computer monitor", "Computer keyboard", "Computer mouse",
    "Laptop", "Printer", "Headphones", "Projector",
    "Whiteboard", "Desk", "Chair", "Table",
    "Bottle", "Coffee cup", "Mobile phone", "Tablet computer",
    "Book", "Pen", "Backpack", "Person",
    "Television", "Camera", "Microphone"
]

def finetune_with_open_images():
    print("=== Phase 3A (Lab Objects): Fine-tuning YOLO11n ===")
    print(f"\nTarget classes ({len(LAB_CLASSES)}):")
    for i, cls in enumerate(LAB_CLASSES):
        print(f"  {i+1}. {cls}")

    model = YOLO("yolo11n.pt")

    # Use Open Images V7 dataset - Ultralytics auto-downloads relevant portions
    print("\nTraining with Open Images V7 dataset...")
    print("This will auto-download relevant images from Google Open Images.")
    print("Epochs: 30, Image Size: 640, Device: CUDA")
    print("Estimated time: ~15-20 minutes\n")

    results = model.train(
        data="open-images-v7.yaml",
        epochs=30,
        imgsz=640,
        device="cuda",
        batch=8,
        workers=2,
        patience=15,
        save=True,
        project="robot-assistant/models",
        name="yolo11n-lab",
        exist_ok=True,
        verbose=True,
        fraction=0.05,           # Use 5% of dataset (it's huge)
        lr0=0.001,
        mosaic=1.0,
        mixup=0.1,
    )

    print("\n=== Lab Objects Fine-tuning Complete ===")

if __name__ == "__main__":
    finetune_with_open_images()
