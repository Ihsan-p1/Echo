from roboflow import Roboflow
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("ROBOFLOW_API_KEY")

if not API_KEY:
    print("Error: ROBOFLOW_API_KEY not found.")
    exit(1)

# Ensure absolute paths for safety
BASE_DIR = os.path.abspath("robot-assistant/data/yolo")
GESTURE_DEST = os.path.join(BASE_DIR, "gestures")
os.makedirs(GESTURE_DEST, exist_ok=True)

print(f"Connecting to Roboflow...")
rf = Roboflow(api_key=API_KEY)
project = rf.workspace("final-year-project-ih9tk").project("hand-gesture-recognition-pgjgg")

print(f"Project found: {project.name}")
print(f"Available versions: {project.versions()}")

# Getting the latest version metadata
version = project.version(1)
print(f"Downloading version 1 to {GESTURE_DEST}...")

# Use yolov8 format as it's well-supported and compatible with yolov11 training
dataset = version.download("yolov8", location=GESTURE_DEST)

print(f"Download complete.")
# Correct way to access classes might vary, let's try a few
try:
    print(f"Classes: {project.classes}")
except:
    try:
        # Some versions have it in version object
        print(f"Classes: {version.project.classes}")
    except:
        print("Could not retrieve classes from project object.")

print(f"Verifying files in {GESTURE_DEST}...")
if os.path.exists(GESTURE_DEST):
    files = os.listdir(GESTURE_DEST)
    print(f"Contents of {GESTURE_DEST}: {files}")
    for item in files:
        subpath = os.path.join(GESTURE_DEST, item)
        if os.path.isdir(subpath):
            print(f"  Subdirectory: {item} contains {os.listdir(subpath)}")
