import time
import json
import subprocess

def query_ollama(prompt, context_objects):
    """Query local Ollama instance with Llama 3.2 3B"""
    objects_str = ", ".join(context_objects)
    
    system_prompt = (
        "You are a friendly robot assistant. "
        "You can see through the camera.\n"
        f"Current visual context: [{objects_str}]\n"
        f"User question: {prompt}\n"
        "Answer in 1-2 sentences. Extract motor commands if any (FORWARD, BACKWARD, LEFT, RIGHT, STOP)."
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
    is_en_reply = ("I " in response_upper) or ("SEE" in response_upper) or ("ROBOT" in response_upper) or ("NOT" in response_upper)
    
    action_match = False
    
    commands = ["FORWARD", "BACKWARD", "LEFT", "RIGHT", "STOP"]
    found_commands = [cmd for cmd in commands if cmd in response_upper]
    
    if expected_action:
        if expected_action.upper() in found_commands:
            action_match = True
    else:
        if not found_commands:
            action_match = True # Passed because no false positives
            
    # Language match check (Basic heuristic)
    lang_match = is_en_reply if lang == "EN" else True
        
    return action_match, lang_match

def test_llm_baseline():
    print("=== TEST 2.3: LLM Baseline Evaluation (Bilingual) ===")
    
    test_prompts = [
        ("What do you see on the table?", ["laptop", "bottle", "book"], "EN", None),
        ("Come to me, robot.", ["person"], "EN", "FORWARD"),
        ("Is it safe to move forward?", [], "EN", None),
        ("Who is in the room?", ["person"], "EN", None),
        ("What objects are in front of you?", ["chair", "table", "bottle"], "EN", None),
        ("Move back a bit.", [], "EN", "BACKWARD"),
        ("Turn left please.", [], "EN", "LEFT"),
        ("Look to the right.", [], "EN", "RIGHT"),
        ("Stop immediately!", [], "EN", "STOP"),
    ]

    stats = {
        "en": {"acc": 0, "latency": 0.0},
        "id": {"acc": 0, "latency": 0.0}
    }
    
    for i, (prompt, context, lang, action) in enumerate(test_prompts):
        lang_key = lang.lower()
        print(f"\n--- Prompt {i+1} [{lang}] ---")
        print(f"Context: {context} | Query: '{prompt}'")
        
        response, latency = query_ollama(prompt, context)
        command_pass, lang_pass = assess_extraction(response, action, lang)
        
        stats[lang_key]["latency"] += float(latency)
        if command_pass and lang_pass:
            stats[lang_key]["acc"] += 1
            
        print(f"Response: {response}")
        print(f"Latency: {latency:.2f}s | Command Extraction: {'PASS' if command_pass else 'FAIL'} | Lang Match: {'PASS' if lang_pass else 'FAIL'}")

    num_en = sum(1 for p in test_prompts if p[2] == "EN")
    num_id = sum(1 for p in test_prompts if p[2] == "ID")

    print("\n=== LLM EVALUATION SUMMARY ===")
    if num_en > 0:
        print(f"EN Score: {stats['en']['acc']}/{num_en} | Avg Latency: {stats['en']['latency']/num_en:.2f}s")
    if num_id > 0:
        print(f"ID Score: {stats['id']['acc']}/{num_id} | Avg Latency: {stats['id']['latency']/num_id:.2f}s")
    
    if (stats['en']['acc'] / 5) < 0.8 or (stats['id']['acc'] / 5) < 0.8:
        print("⚠️ FINE-TUNE TRIGGERED: Command extraction < 80% or Language mismatch > 20%")
    else:
        print("✅ LLM baseline passed.")

if __name__ == "__main__":
    test_llm_baseline()
