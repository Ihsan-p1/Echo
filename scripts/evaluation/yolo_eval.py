import os
import time
from ultralytics import YOLO

def download_images(image_urls, dest_dir="robot-assistant/data/yolo/eval_images"):
    import requests
    os.makedirs(dest_dir, exist_ok=True)
    images = []
    
    print(f"Downloading {len(image_urls)} sample images...")
    for i, url in enumerate(image_urls):
        try:
            target_path = os.path.join(dest_dir, f"sample_{i+1}.jpg")
            if not os.path.exists(target_path):
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                with open(target_path, 'wb') as f:
                    f.write(response.content)
            images.append(target_path)
        except Exception as e:
            print(f"Failed to download {url}: {e}")
    return images

def test_yolo_baseline():
    print("=== TEST 2.1: YOLO Baseline Evaluation ===")
    
    # 10 test image URLs representing desk/indoor objects
    # (laptop, cup, person, bottle, TV, keyboard, cell phone)
    sample_urls = [
        "https://images.unsplash.com/photo-1517336714731-489689fd1ca8?w=640", # Laptop + person
        "https://images.unsplash.com/photo-1499951360447-b19be8fe80f5?w=640", # Desk, laptop, cup
        "https://images.unsplash.com/photo-1527443224154-c4a3942d3acf?w=640", # Desk, monitor, keyboard
        "https://images.unsplash.com/photo-1542744173-8e7e53415bb0?w=640", # Meeting, persons, laptops
        "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=640", # Headphones, objects
        "https://images.unsplash.com/photo-1583394838336-acd977736f90?w=640", # Headphones on desk
        "https://images.unsplash.com/photo-1593642702821-c823b2816915?w=640", # Laptop on desk
        "https://images.unsplash.com/photo-1621252179027-94459d278660?w=640", # Water bottle
        "https://images.unsplash.com/photo-1512499616092-2023a8511470?w=640", # Person sitting
        "https://images.unsplash.com/photo-1525547719571-a2d4ac8945e2?w=640"  # Person using laptop
    ]
    
    images = download_images(sample_urls)
    if not images:
        print("Failed to download sample images.")
        return
        
    print("\nLoading YOLO11n on CUDA...")
    model = YOLO("yolo11n.pt")
    
    total_time = 0
    total_detections = 0
    results_summary = []
    
    print("\nRunning Inference...")
    for img_path in images:
        start_time = time.time()
        # running inference on image
        # verbose=False to keep logs clean
        results = model(img_path, device='cuda', verbose=False)
        inference_time = (time.time() - start_time) * 1000  # ms
        
        boxes = results[0].boxes
        num_objects = len(boxes)
        # Getting the names using model.names list matching the predicted classes
        detected_names = [model.names[int(cls)] for cls in boxes.cls]
        
        total_time += inference_time
        total_detections += num_objects
        
        summary = f"Image: {os.path.basename(img_path)} | Time: {inference_time:.1f}ms | Objects ({num_objects}): {', '.join(detected_names)}"
        print(summary)
        results_summary.append(summary)
        
    avg_inference_time = total_time / len(images)
    print("\n=== YOLO EVALUATION SUMMARY ===")
    print(f"Total Images: {len(images)}")
    print(f"Total Objects Detected: {total_detections}")
    print(f"Average Inference Time: {avg_inference_time:.2f}ms")
    
    if avg_inference_time > 100:
        print("⚠️ FINE-TUNE TRIGGERED: Average inference time > 100ms")
    else:
        print("✅ Inference time within limits (< 100ms)")
        
    # Note: Precise mAP requires ground truth labels which we don't have
    # for these random unsplash images. As a proxy, we verify it detected objects.
    if total_detections < len(images):
        print("⚠️ FINE-TUNE TRIGGERED: False negative rate too high (mAP proxy failed)")
    else:
        print("✅ Object detection confidence proxy passed.")

if __name__ == "__main__":
    test_yolo_baseline()
