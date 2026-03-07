import time
import json
import subprocess

def query_ollama(prompt, context_objects):
    """Query local Ollama instance with Llama 3.2 3B"""
    objects_str = ", ".join(context_objects)
    
    system_prompt = (
        "Kamu adalah robot asisten yang ramah. "
        "Kamu dapat melihat melalui kamera.\n"
        f"Context visual saat ini: [{objects_str}]\n"
        f"Pertanyaan user: {prompt}\n"
        "Jawab dalam 1-2 kalimat. Ekstrak motor command jika ada (MAJU, MUNDUR, KIRI, KANAN, STOP)."
    )
    
    # Using ollama CLI for direct, simple interaction without depending on extra libraries
    cmd = ['ollama', 'run', 'llama3.2:3b', system_prompt]
    
    start_time = time.time()
    try:
        # Run subprocess and capture output
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15, encoding="utf-8")
        response_text = result.stdout.strip()
    except subprocess.TimeoutExpired:
        response_text = "[TIMEOUT]"
    except Exception as e:
        response_text = f"[ERROR: {str(e)}]"
        
    latency = time.time() - start_time
    return response_text, latency

def assess_extraction(response_text, expected_action, lang):
    """Simple heuristic to check command extraction"""
    response_upper = response_text.upper()
    is_id_reply = ("SAYA" in response_upper) or ("MELIHAT" in response_upper) or ("ROBOT" in response_upper) or ("TIDAK" in response_upper)
    
    has_command = False
    action_match = False
    
    commands = ["MAJU", "MUNDUR", "KIRI", "KANAN", "STOP"]
    found_commands = [cmd for cmd in commands if cmd in response_upper]
    
    if expected_action:
        if expected_action.upper() in found_commands:
            action_match = True
    else:
        if not found_commands:
            action_match = True # Passed because no false positives
            
    # Language match check (Basic heuristic)
    lang_match = False
    if lang == "ID" and is_id_reply:
         lang_match = True
    elif lang == "EN" and not is_id_reply:
         lang_match = True
    elif lang == "ID" and not is_id_reply and ("I " in response_text or "SEE " in response_text):
        lang_match = False
    else:
        lang_match = True # Benefit of doubt
        
    return action_match, lang_match

def test_llm_baseline():
    print("=== TEST 2.3: LLM Baseline Evaluation (Bilingual) ===")
    
    test_prompts = [
        # --- English ---
        ("What do you see on the table?", ["laptop", "bottle", "book"], "EN", None),
        ("Come to me, robot.", ["person"], "EN", "MAJU"),
        ("Is it safe to move forward?", [], "EN", None),
        ("Who is in the room?", ["person"], "EN", None),
        ("What objects are in front of you?", ["chair", "table", "bottle"], "EN", None),
        # --- Indonesian ---
        ("Ada apa di mejaku?", ["laptop", "gelas", "buku"], "ID", None),
        ("Robot, datang ke saya.", ["person"], "ID", "MAJU"),
        ("Apakah aman untuk maju?", [], "ID", None),
        ("Siapa yang ada di ruangan ini?", ["person"], "ID", None),
        ("Apa yang kamu lihat?", ["kursi", "meja", "botol"], "ID", None),
    ]

    en_acc = 0
    id_acc = 0
    en_latex = 0
    id_latex = 0
    
    for i, (prompt, context, lang, action) in enumerate(test_prompts):
        print(f"\n--- Prompt {i+1} [{lang}] ---")
        print(f"Context: {context} | Query: '{prompt}'")
        
        response, latency = query_ollama(prompt, context)
        
        command_pass, lang_pass = assess_extraction(response, action, lang)
        
        if lang == "EN":
            en_latex += latency
            if command_pass and lang_pass: en_acc += 1
        else:
            id_latex += latency
            if command_pass and lang_pass: id_acc += 1
            
        print(f"Response: {response}")
        print(f"Latency: {latency:.2f}s | Command Extraction: {'PASS' if command_pass else 'FAIL'} | Lang Match: {'PASS' if lang_pass else 'FAIL'}")

    print("\n=== LLM EVALUATION SUMMARY ===")
    print(f"EN Score: {en_acc}/5 | Avg Latency: {en_latex/5:.2f}s")
    print(f"ID Score: {id_acc}/5 | Avg Latency: {id_latex/5:.2f}s")
    
    if (en_acc / 5) < 0.8 or (id_acc / 5) < 0.8:
        print("⚠️ FINE-TUNE TRIGGERED: Command extraction < 80% or Language mismatch > 20%")
    else:
        print("✅ LLM baseline passed.")

if __name__ == "__main__":
    test_llm_baseline()
