import cv2
import time
from ultralytics import YOLO

def test_finetuned_webcam():
    print("Memuat model YOLO11n fine-tuned (Campus)...")
    model = YOLO("robot-assistant/models/yolo11n-campus.pt")
    
    print("Membuka kamera laptop (Webcam)...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Tidak dapat membuka webcam.")
        return

    print("Kamera berhasil dibuka. Tekan 'q' untuk keluar.")
    
    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        current_time = time.time()
        results = model(frame, device='cuda', verbose=False, conf=0.5)
        
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time
        
        annotated_frame = results[0].plot()
        
        cv2.putText(annotated_frame, f"FPS: {fps:.1f} | LAB & CAMPUS", (10, 30), 
                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow("YOLO LAB & CAMPUS DETECTOR", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_finetuned_webcam()
