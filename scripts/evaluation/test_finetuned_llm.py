import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import os

def test_llm_adapter():
    print("=== Phase 3.3: Testing Fine-Tuned LLM Adapter ===")
    
    model_id = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
    adapter_path = "robot-assistant/models/llm-finetuned-adapter"
    
    if not os.path.exists(adapter_path):
        print(f"Error: Adapter path {adapter_path} not found.")
        return

    print("Loading 4-bit base model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    print("Applying LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    test_cases = [
        {"context": ["person", "chair"], "query": "Move forward past the chair."},
        {"context": ["laptop", "bottle"], "query": "Stop! There is a bottle in your way."},
        {"context": [], "query": "Turn left 90 degrees."},
        {"context": ["table"], "query": "What do you see?"}
    ]

    print("\nRunning test cases...")
    for i, case in enumerate(test_cases):
        prompt = f"### Instruction: You are a friendly robot assistant. You can see through the camera. Analyze the visual context and respond to the user. Extract a motor command if needed (FORWARD, BACKWARD, LEFT, RIGHT, STOP, NONE).\n### Input: Visual Context: {case['context']}\nUser Query: {case['query']}\n### Response: "
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the response part
        if "### Response: " in response:
            final_resp = response.split("### Response: ")[1].strip()
        else:
            final_resp = response
            
        print(f"\nTest {i+1}:")
        print(f"  Query: {case['query']}")
        print(f"  Result:\n{final_resp}")

if __name__ == "__main__":
    test_llm_adapter()
