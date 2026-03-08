import os
import time
from faster_whisper import WhisperModel
from gtts import gTTS

def generate_sentence_audio(text, lang, filename):
    try:
        tts = gTTS(text, lang=lang)
        tts.save(filename)
        return filename
    except Exception as e:
        print(f"Error generating {filename}: {e}")
        return None

def test_stt_baseline():
    print("=== TEST 2.2: STT Baseline Evaluation (Bilingual) ===")
    
    sentences = [
        ("Robot, what do you see in front of you?", "en"),
        ("Is it safe to move forward?", "en"),
        ("Come here, robot.", "en"),
        ("What objects are on the table?", "en"),
        ("Stop moving.", "en"),
        ("Robot, look ahead.", "en"),
        ("What's on your table?", "en"),
        ("Is the path clear?", "en"),
        ("Move forward please.", "en"),
        ("Halt all movements.", "en")
    ]
    
    os.makedirs("robot-assistant/data/whisper/eval_audio", exist_ok=True)
    audio_files = []
    
    print("\nGenerating baseline audio files...")
    for i, (text, lang) in enumerate(sentences):
        filename = f"robot-assistant/data/whisper/eval_audio/sentence_{i}.wav"
        if not os.path.exists(filename):
            generate_sentence_audio(text, lang, filename)
        audio_files.append((filename, text, lang))
        
    print("\nLoading faster-whisper medium on CUDA...")
    model = WhisperModel("medium", device="cuda", compute_type="float16")
    
    results_summary = []
    test_failures = 0
    total_latency_en = 0
    total_latency_id = 0
    
    print("\nRunning Inference...")
    for filename, ground_truth, expected_lang in audio_files:
        if not os.path.exists(filename):
            continue
            
        start_time = time.time()
        segments, info = model.transcribe(filename, beam_size=5)
        
        # consume generator to get transcript text
        transcript = "".join([segment.text for segment in segments]).strip()
        latency = time.time() - start_time
        
        # Metrics logic
        passed_lang = (info.language == expected_lang)
        passed_latency = (latency < 2.0)
        
        if expected_lang == "en":
            total_latency_en += latency
        else:
            total_latency_id += latency
            
        if not passed_lang or not passed_latency:
            test_failures += 1
            
        status = "✅" if passed_lang and passed_latency else "❌"
        summary = f"{status} [{expected_lang.upper()}] | Detected Lang: {info.language} | Time: {latency:.2f}s\n  Ground: \"{ground_truth}\"\n  Result: \"{transcript}\""
        print(summary)
        results_summary.append(summary)
        
    avg_lat_en = total_latency_en / 10 if len(audio_files) > 0 else 0
    
    print("\n=== STT EVALUATION SUMMARY ===")
    print(f"Average Latency (EN): {float(avg_lat_en):.2f}s")
    
    # Simple heuristic trigger for fine-tuning based on test cases passed
    if test_failures > 0 or avg_lat_en > 2.0:
        print(f"⚠️ FINE-TUNE TRIGGERED: 1 or more condition failed (Failures: {test_failures}). Latency > 2s or Language Confusion.")
    else:
        print("✅ STT baseline passed.")

if __name__ == "__main__":
    test_stt_baseline()
