import requests
import os

def download_file(url, target_path):
    print(f"Downloading {url} to {target_path}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(target_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded {target_path}")

base_dir = "robot-assistant/voices"
os.makedirs(base_dir, exist_ok=True)

onnx_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main/id/id_ID/news_tts/medium/id_ID-news_tts-medium.onnx"
json_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main/id/id_ID/news_tts/medium/id_ID-news_tts-medium.onnx.json"

download_file(onnx_url, os.path.join(base_dir, "id_ID-news_tts-medium.onnx"))
download_file(json_url, os.path.join(base_dir, "id_ID-news_tts-medium.onnx.json"))
