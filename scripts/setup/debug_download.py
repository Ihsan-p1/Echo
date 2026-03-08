from roboflow import Roboflow
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("ROBOFLOW_API_KEY")

rf = Roboflow(api_key=API_KEY)
# We use the user-provided project
project = rf.workspace("final-year-project-ih9tk").project("hand-gesture-recognition-pgjgg")
version = project.version(1)

print("Starting default download (no location specified)...")
# Defaulting location to see if it appears in CWD
dataset = version.download("yolov8") 
print(f"Download call finished.")

print("Checking CWD for updates...")
cwd_items = os.listdir(".")
for item in cwd_items:
    if os.path.isdir(item) and ("hand" in item.lower() or "gesture" in item.lower()):
        print(f"FOUND POTENTIAL FOLDER: {item}")
        print(f"Contents: {os.listdir(item)}")
