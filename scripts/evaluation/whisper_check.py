from faster_whisper import WhisperModel
import time

def check_whisper():
    print("Loading faster-whisper medium model...")
    start_time = time.time()
    
    # Load model (uses FP16 if CUDA is available)
    model = WhisperModel("medium", device="cuda", compute_type="float16")
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds.")
    
    print("\nTranscribing 'test_audio.wav'...")
    start_time = time.time()
    segments, info = model.transcribe("test_audio.wav", language="id")
    
    # Print results
    print(f"Detected language: '{info.language}' with probability {info.language_probability}")
    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
        
    transcription_time = time.time() - start_time
    print(f"Transcription completed in {transcription_time:.2f} seconds.")

if __name__ == "__main__":
    check_whisper()
