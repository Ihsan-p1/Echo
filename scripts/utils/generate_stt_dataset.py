import os
import json
from gtts import gTTS
import random

def generate_stt_dataset():
    print("=== Phase 3B: Generating STT Dataset (English) ===")
    
    base_dir = "robot-assistant/data/whisper/dataset"
    audio_dir = os.path.join(base_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    
    # Core Robot Intentions
    intents = {
        "forward": [
            "move forward", "go forward", "onward", "move ahead", 
            "proceed forward", "walk forward", "advance", "step forward",
            "robot move forward", "please move forward", "move forward now"
        ],
        "backward": [
            "move backward", "go back", "retreat", "step back", 
            "reverse", "move behind", "go backwards", "robot go back",
            "back up a bit", "please retreat"
        ],
        "left": [
            "turn left", "rotate left", "go left", "look left", 
            "move left", "spin to the left", "robot turn left"
        ],
        "right": [
            "turn right", "rotate right", "go right", "look right", 
            "move right", "spin to the right", "robot turn right"
        ],
        "stop": [
            "stop", "halt", "stay", "freeze", "don't move", 
            "stop moving", "cease all movement", "wait here", 
            "robot stop", "emergency stop", "terminate movement"
        ],
        "query": [
            "what do you see?", "identify objects", "what's in front?",
            "is it safe?", "check surroundings", "look around",
            "scan the room", "are there any people?", "detect objects",
            "tell me what's ahead", "do you see anything?"
        ],
        "specific_obj": [
            "where is the laptop?", "find the bottle", "locate the chair",
            "do you see a keyboard?", "point to the monitor",
            "where is the mouse?", "identify the desk"
        ]
    }

    # Templates for natural language variations
    templates = [
        "{cmd}",
        "robot, {cmd}",
        "could you {cmd}?",
        "please {cmd}",
        "hey robot, {cmd} now",
        "i want you to {cmd}",
        "can you {cmd} please?"
    ]
    
    # Diverse English variations (TLDs)
    tlds = ['com', 'co.uk', 'com.au', 'co.in', 'ca', 'ie', 'co.za']
    
    dataset_metadata = []
    print(f"Generating audio files in {audio_dir}...")
    
    count = 0
    generated_texts = set()

    for intent, cmds in intents.items():
        for base_cmd in cmds:
            for template in templates:
                cmd_text = template.format(cmd=base_cmd)
                # Avoid exact duplicate texts if they happen
                if cmd_text in generated_texts: continue
                generated_texts.add(cmd_text)
                
                # Pick a random voice for this variation
                tld = random.choice(tlds)
                filename = f"cmd_{count:03d}.wav"
                filepath = os.path.join(audio_dir, filename)
                
                try:
                    tts = gTTS(text=cmd_text, lang='en', tld=tld, slow=random.choice([False, False, True]))
                    tts.save(filepath)
                    
                    dataset_metadata.append({
                        "audio_filepath": filename,
                        "text": cmd_text.lower()
                    })
                    count += 1
                    
                    if count % 25 == 0:
                        print(f"Generated {count} files...")
                    
                    # Limit to ~500 samples for a balanced, faster fine-tuning
                    if count >= 600:
                        break
                except Exception as e:
                    print(f"Failed to generate {filename}: {e}")
            if count >= 600: break
        if count >= 600: break
                
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
