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

DEST = "robot-assistant/data/yolo/gadgets"
os.makedirs(DEST, exist_ok=True)

rf = Roboflow(api_key=API_KEY) if API_KEY else None

if not rf:
    exit(1)

# Try downloading gadget/electronics datasets from Roboflow Universe
# These contain computer lab objects: monitor, keyboard, mouse, headphones, etc.
datasets_to_try = [
    ("daniel-voskergian", "electronics-object-detection", 1),
    ("new-workspace-cdxqy", "gadget-sjncl", 2),
    ("iot-fwsxb", "gadget-n6gye", 1),
]

for workspace, project_name, version_num in datasets_to_try:
    try:
        print(f"\nTrying: {workspace}/{project_name}/v{version_num}")
        project = rf.workspace(workspace).project(project_name)
        version = project.version(version_num)
        dataset = version.download("yolov8", location=DEST)
        print(f"Downloaded successfully to {DEST}")
        print(f"Classes: {version.classes}")
        break
    except Exception as e:
        print(f"Failed: {e}")
        continue
