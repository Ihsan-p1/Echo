import os
import json
import random

def generate_llm_dataset():
    print("=== Phase 3C: Generating LLM Dataset (Bilingual + Command) ===")
    
    base_dir = "robot-assistant/data/llm"
    os.makedirs(base_dir, exist_ok=True)
    out_path = os.path.join(base_dir, "robot_commands.jsonl")
    
    dataset = []
    
    # Templates for generating data
    # format: (instruction, input_objects, output_response)
    
    templates = [
        # English
        ("What do you see on the table?", ["laptop", "bottle", "book"], "I see a laptop, a bottle, and a book on the table.\nCOMMAND: NULL"),
        ("What's in front of you?", ["monitor", "keyboard", "mouse"], "I can see a monitor, a keyboard, and a mouse in front of me.\nCOMMAND: NULL"),
        ("Come to me, robot.", ["person"], "Understood, I am coming to you now.\nCOMMAND: MAJU"),
        ("Come here.", ["person", "chair"], "I see you. Proceeding towards your location.\nCOMMAND: MAJU"),
        ("Is it safe to move forward?", [], "Yes, the path appears to be clear. I can move forward.\nCOMMAND: NULL"),
        ("Is it safe to move forward?", ["person"], "I see someone blocking the path. It is not currently safe to move forward.\nCOMMAND: STOP"),
        ("Who is in the room?", ["person"], "I see a person in the room with me.\nCOMMAND: NULL"),
        ("Stop moving right now.", ["wall"], "Understood. Stopping immediately.\nCOMMAND: STOP"),
        ("Turn to the left.", [], "Turning left as requested.\nCOMMAND: KIRI"),
        ("Turn right.", ["chair"], "Turning right now.\nCOMMAND: KANAN"),
        ("Go back securely.", [], "Moving backwards.\nCOMMAND: MUNDUR"),
        ("What objects are in front of you?", ["chair", "table", "bottle"], "I detect a chair, a table, and a bottle.\nCOMMAND: NULL"),

        # Indonesian
        ("Ada apa di mejaku?", ["laptop", "gelas", "buku"], "Saya melihat sebuah laptop, gelas, dan buku di mejamu.\nCOMMAND: NULL"),
        ("Kamu lihat apa di depan?", ["monitor", "keyboard", "mouse"], "Di depan saya ada monitor, keyboard, dan mouse.\nCOMMAND: NULL"),
        ("Robot, datang ke saya.", ["person"], "Baik, saya akan bergerak ke arah Anda.\nCOMMAND: MAJU"),
        ("Maju ke sini.", ["person", "chair"], "Saya melihat Anda. Saya segera ke sana.\nCOMMAND: MAJU"),
        ("Apakah aman untuk maju?", [], "Jalur di depan terlihat kosong dan aman untuk maju.\nCOMMAND: NULL"),
        ("Apakah aman untuk maju?", ["person"], "Ada seseorang di depan. Saat ini tidak aman untuk maju.\nCOMMAND: STOP"),
        ("Siapa yang ada di ruangan ini?", ["person"], "Saya melihat ada seseorang di dalam ruangan ini.\nCOMMAND: NULL"),
        ("Berhenti sekarang juga.", ["wall"], "Baik, saya berhenti bergerak agar tidak menabrak.\nCOMMAND: STOP"),
        ("Belok kiri.", [], "Saya berbelok ke kiri.\nCOMMAND: KIRI"),
        ("Belok ke kanan.", ["chair"], "Baik, berbelok ke kanan.\nCOMMAND: KANAN"),
        ("Mundur perlahan.", [], "Saya bergerak mundur.\nCOMMAND: MUNDUR"),
        ("Apa yang kamu lihat?", ["kursi", "meja", "botol"], "Saya bisa melihat kursi, meja, dan sebuah botol.\nCOMMAND: NULL"),
    ]
    
    # Generate variations to reach ~100+ items
    print(f"Generating synthetic dataset from {len(templates)} base templates...")
    
    for i in range(120):
        # Pick a random template
        inst, objs, evt = random.choice(templates)
        
        # Slightly modify the object string occasionally
        if len(objs) > 0 and random.random() > 0.5:
            random.shuffle(objs)
            obj_str = "objects: " + ", ".join(objs)
        elif len(objs) > 0:
            obj_str = "objects: " + ", ".join(objs)
        else:
            obj_str = "objects: none"
            
        dataset.append({
            "instruction": inst,
            "input": obj_str,
            "output": evt
        })

    with open(out_path, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item) + '\n')
            
    print(f"Done! Generated {len(dataset)} instruction pairs.")
    print(f"File saved to {out_path}")
    print("\nFormat ready for Unsloth / HuggingFace fine-tuning.")

if __name__ == "__main__":
    generate_llm_dataset()
