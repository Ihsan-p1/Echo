import os
import json
from gtts import gTTS
import random

def generate_stt_dataset():
    print("=== Phase 3B: Generating STT Dataset (English) ===")
    
    base_dir = "robot-assistant/data/whisper/dataset"
    audio_dir = os.path.join(base_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    
    # English commands for the robot
    commands = [
        "Robot, what do you see in front of you?",
        "Is it safe to move forward?",
        "Come here, robot.",
        "What objects are on the table?",
        "Stop moving.",
        "Turn left.",
        "Turn right.",
        "Go back.",
        "Move forward.",
        "What is that object?",
        "Tell me what you see.",
        "Look around the room.",
        "Follow me.",
        "Are there any people here?",
        "Can you see a laptop?",
        "Where is the bottle?",
        "Stop immediately.",
        "Rotate ninety degrees.",
        "Move closer to the table.",
        "Wait here."
    ]
    
    # We will generate a few variations for each command to create a small dataset
    variations = ["en-us", "en-uk", "en-au", "en-in"] # Different English accents
    
    dataset_metadata = []
    
    print(f"Generating audio files in {audio_dir}...")
    
    count = 0
    for cmd in commands:
        for tld in ['com', 'co.uk', 'com.au', 'co.in']: # Using top-level domains to simulate accents in gTTS
            filename = f"cmd_{count:03d}.wav"
            filepath = os.path.join(audio_dir, filename)
            
            try:
                # Generate audio
                tts = gTTS(text=cmd, lang='en', tld=tld, slow=random.choice([False, False, True]))
                tts.save(filepath)
                
                # Save metadata (ground truth)
                dataset_metadata.append({
                    "audio_filepath": filename,
                    "text": cmd.lower() # Whisper training usually lowercases
                })
                count += 1
                
                if count % 10 == 0:
                    print(f"Generated {count} files...")
                    
            except Exception as e:
                print(f"Failed to generate {filename}: {e}")
                
    # Save metadata to JSONL for HuggingFace datasets compatibility
    metadata_path = os.path.join(base_dir, "metadata.jsonl")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        for item in dataset_metadata:
            f.write(json.dumps(item) + '\n')
            
    print(f"\nDone! Generated {count} audio files.")
    print(f"Metadata saved to {metadata_path}")
    print("\nFormat ready for HuggingFace / OpenAI Whisper fine-tuning.")

if __name__ == "__main__":
    generate_stt_dataset()
