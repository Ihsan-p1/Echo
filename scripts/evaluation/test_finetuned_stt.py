import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel
import librosa
import os
import time

def test_finetuned_whisper():
    print("=== FINAL TEST: Fine-Tuned Whisper (LoRA) Evaluation ===")
    
    model_id = "openai/whisper-small"
    adapter_path = "robot-assistant/models/whisper-finetuned-best"
    eval_dir = "robot-assistant/data/whisper/eval_audio"
    
    # Ground truth from stt_eval.py
    ground_truth = [
        "Robot, what do you see in front of you?",
        "Is it safe to move forward?",
        "Come here, robot.",
        "What objects are on the table?",
        "Stop moving.",
        "Robot, look ahead.",
        "What's on your table?",
        "Is the path clear?",
        "Move forward please.",
        "Halt all movements."
    ]
    
    if not os.path.exists(adapter_path):
        print(f"Error: Fine-tuned weights not found at {adapter_path}")
        return

    print(f"Loading base model: {model_id}...")
    model = WhisperForConditionalGeneration.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)
    processor = WhisperProcessor.from_pretrained(adapter_path)
    
    print(f"Applying LoRA adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    
    print("\nRunning Inference on Evaluation Set...")
    total_latency = 0
    correct_transcripts = 0
    results_found = 0
    
    for i, truth in enumerate(ground_truth):
        audio_path = os.path.join(eval_dir, f"sentence_{i}.wav")
        if not os.path.exists(audio_path):
            print(f"File not found: {audio_path}")
            continue
            
        results_found += 1
        start_time = time.time()
        audio, sr = librosa.load(audio_path, sr=16000)
        input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to("cuda").to(torch.float16)
        
        with torch.no_grad():
            # Force English language for transcription
            forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
            predicted_ids = model.generate(input_features, max_length=225, forced_decoder_ids=forced_decoder_ids)
            
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        latency = time.time() - start_time
        total_latency += latency
        
        # Clean strings for comparison
        clean_model = transcription.lower().strip(".,?!")
        clean_truth = truth.lower().strip(".,?!")
        
        if clean_model == clean_truth:
            status = "✅"
            correct_transcripts += 1
        else:
            status = "❌"
            
        print(f"{status} Sample {i} | Time: {latency:.2f}s")
        print(f"  Truth: \"{truth}\"")
        print(f"  Model: \"{transcription}\"")
        
    if results_found > 0:
        avg_latency = total_latency / results_found
        accuracy = (correct_transcripts / results_found) * 100
        
        print("\n=== FINAL STT EVALUATION SUMMARY ===")
        print(f"Accuracy: {accuracy:.1f}% ({correct_transcripts}/{results_found})")
        print(f"Average Latency: {avg_latency:.2f}s")
    else:
        print("\nNo evaluation files were processed.")
    
    print("Fine-tuning verification complete.")

if __name__ == "__main__":
    test_finetuned_whisper()
