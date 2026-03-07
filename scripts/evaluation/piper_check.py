from piper.voice import PiperVoice
import wave

def check_piper():
    print("Loading piper TTS model (id_ID-news_tts-medium)...")
    try:
        voice = PiperVoice.load(
            "robot-assistant/voices/id_ID-news_tts-medium.onnx",
            config_path="robot-assistant/voices/id_ID-news_tts-medium.onnx.json"
        )
        print("Piper model loaded successfully.")
        
        output_file = "test_tts_output.wav"
        print(f"Synthesizing test audio to {output_file}...")
        with wave.open(output_file, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(voice.config.sample_rate)
            voice.synthesize("Halo, saya robot asisten yang siap membantu.", f)
        print("Success! TTS file generated.")
    except Exception as e:
        print(f"Error loading or synthesizing Piper TTS: {e}")

if __name__ == "__main__":
    check_piper()
