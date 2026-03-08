import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel, PeftConfig
import librosa
import os

def verify_finetuning():
    print("=== Verifying Fine-Tuned Whisper (LoRA) ===")
    
    model_id = "openai/whisper-medium"
    adapter_path = "robot-assistant/models/whisper-finetuned-best"
    audio_path = "robot-assistant/data/temp/test_audio.wav"
    
    if not os.path.exists(adapter_path):
        print(f"Error: Adapter path {adapter_path} not found.")
        return

    print(f"Loading base model: {model_id}...")
    model = WhisperForConditionalGeneration.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)
    processor = WhisperProcessor.from_pretrained(adapter_path)
    
    print(f"Loading LoRA adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    
    if os.path.exists(audio_path):
        print(f"Transcribing {audio_path}...")
        audio, sr = librosa.load(audio_path, sr=16000)
        input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to("cuda").to(torch.float16)
        
        with torch.no_grad():
            predicted_ids = model.generate(input_features)
            
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        print(f"\nResult: \"{transcription}\"")
    else:
        print(f"Warning: Audio file {audio_path} not found. Skipping transcription test.")

if __name__ == "__main__":
    verify_finetuning()
