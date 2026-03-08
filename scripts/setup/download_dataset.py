from roboflow import Roboflow
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API_KEY should be set as an environment variable or in a .env file
API_KEY = os.getenv("ROBOFLOW_API_KEY") 

if not API_KEY:
    print("Error: ROBOFLOW_API_KEY environment variable not set.")
    # For local testing, you can temporarily hardcode it here, but DO NOT COMMIT IT.
    # API_KEY = "..."

# Target directories
GADGET_DEST = "robot-assistant/data/yolo/gadgets"
GESTURE_DEST = "robot-assistant/data/yolo/gestures"
os.makedirs(GADGET_DEST, exist_ok=True)
os.makedirs(GESTURE_DEST, exist_ok=True)

rf = Roboflow(api_key=API_KEY) if API_KEY else None

if rf is None:
    print("Error: Roboflow object not initialized. Check your API key.")
    exit(1)

# Hand Gesture Dataset (for control)
gesture_project = ("final-year-project-ih9tk", "hand-gesture-recognition-pgjgg", 1)

# Lab Objects Datasets (optional fallback/secondary)
gadget_projects = [
    ("daniel-voskergian", "electronics-object-detection", 1),
    ("new-workspace-cdxqy", "gadget-sjncl", 2),
]

if rf is not None:
    # 1. Download Gestures
    try:
        w, p, v = gesture_project
        print(f"\nDownloading GESTURES: {w}/{p}/v{v}")
        project = rf.workspace(w).project(p)
        version = project.version(v)
        dataset = version.download("yolov11", location=GESTURE_DEST) # Use yolov11 if possible, fallback to yolov8
        print(f"Downloaded Gestures successfully to {GESTURE_DEST}")
        print(f"Classes: {version.classes}")
    except Exception as e:
        print(f"Gesture Download Failed: {e}")

    # 2. Download Gadgets (if needed)
    for workspace, project_name, version_num in gadget_projects:
        try:
            print(f"\nTrying GADGETS: {workspace}/{project_name}/v{version_num}")
            project = rf.workspace(workspace).project(project_name)
            version = project.version(version_num)
            dataset = version.download("yolov8", location=GADGET_DEST)
            print(f"Downloaded successfully to {GADGET_DEST}")
            print(f"Classes: {version.classes}")
            break
        except Exception as e:
            print(f"Failed: {e}")
            continue
