from ultralytics import YOLO

def main():
    model = YOLO('yolo11n.pt')
    results = model('https://ultralytics.com/images/bus.jpg', device='cuda')
    print("Detected boxes:")
    print(results[0].boxes)

if __name__ == "__main__":
    main()
