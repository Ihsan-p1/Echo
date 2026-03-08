import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

def finetune_llm():
    print("=== Phase 3.3: Fine-Tuning Llama 3.2 3B (Robot Assistant) ===")
    
    dataset_path = "robot-assistant/data/llm/finetune_data.json"
    model_id = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit" # Using a pre-quantized version for faster/easier loading
    output_dir = "robot-assistant/models/llm-finetuned-adapter"
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return

    # 1. Load Dataset
    print("Loading dataset...")
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    # 2. BitsAndBytes Config (4-bit quantization)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, # Use bfloat16 for Ampere GPU (RTX 30 series)
        bnb_4bit_use_double_quant=True,
    )

    # 3. Load Model & Tokenizer
    print(f"Loading base model: {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 4. LoRA Config
    print("Preparing LoRA Config...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 5. Training Arguments
    print("Starting Training (Optimized for 6GB VRAM)...")
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        max_steps=100,
        logging_steps=10,
        save_steps=50,
        fp16=False,
        bf16=True, # RTX 3050 supports BF16
        optim="paged_adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
        max_length=512, # Correct parameter name for this version
        dataset_text_field="text"
    )

    # 6. Manual Formatting
    def format_dataset(example):
        example["text"] = f"### Instruction: {example['instruction']}\n### Input: {example['input']}\n### Response: {example['output']}"
        return example

    dataset = dataset.map(format_dataset)

    # 7. Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_args
    )

    trainer.train()

    # 7. Save the Adapter
    print(f"Saving fine-tuned adapter to {output_dir}...")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Training Complete!")

if __name__ == "__main__":
    finetune_llm()
