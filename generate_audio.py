from gtts import gTTS
import os

def generate_test_audio():
    text = "Halo, ini adalah tes transkripsi"
    tts = gTTS(text, lang="id")
    tts.save("test_audio.wav")
    print("Generated test_audio.wav")

if __name__ == "__main__":
    generate_test_audio()
