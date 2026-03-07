import requests

url = "https://huggingface.co/api/models/rhasspy/piper-voices/tree/main/id/id_ID/news_tts"
try:
    response = requests.get(url)
    response.raise_for_status()
    print("Available directories under id_ID:")
    for item in response.json():
        print(item)
except Exception as e:
    print("Error:", e)
