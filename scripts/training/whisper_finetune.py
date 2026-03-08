"""
Fine-tune openai/whisper-medium using PEFT (LoRA) on the custom synthetic dataset.
This script loads the JSONL metadata, processes audio, and trains the model.
"""
import os
import torch
from datasets import load_dataset, Audio
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import evaluate

def finetune_whisper():
    print("=== Phase 3B: Fine-Tuning Whisper (English Commands) ===")
    
    data_dir = "robot-assistant/data/whisper/dataset"
    metadata_path = os.path.join(data_dir, "metadata.jsonl")
    
    if not os.path.exists(metadata_path):
        print(f"Error: Dataset not found at {metadata_path}")
        return
        
    print("Loading dataset...")
    # Load dataset from JSONL mapping audio files to text
    dataset = load_dataset("json", data_files=metadata_path, split="train")
    
    # Fix paths to be absolute based on data_dir
    def fix_audio_path(batch):
        batch["audio"] = os.path.join(data_dir, "audio", batch["audio_filepath"])
        return batch
        
    dataset = dataset.map(fix_audio_path)
    
    import librosa
    
    # We will split a small validation set before processing
    dataset = dataset.train_test_split(test_size=0.1)
    
    print("\nLoading Whisper Processor & Tokenizer...")
    model_id = "openai/whisper-small"
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)
    tokenizer = WhisperTokenizer.from_pretrained(model_id, language="English", task="transcribe")
    processor = WhisperProcessor.from_pretrained(model_id, language="English", task="transcribe")
    
    # Preprocessing function using librosa directly
    def prepare_dataset(batch):
        # load and resample audio data using librosa
        audio_path = batch["audio"]
        audio_array, sampling_rate = librosa.load(audio_path, sr=16000)
        
        # compute log-Mel input features
        batch["input_features"] = feature_extractor(audio_array, sampling_rate=sampling_rate).input_features[0]
        # encode target text to label ids
        batch["labels"] = tokenizer(batch["text"]).input_ids
        return batch
        
    print("Extracting features (this may take a moment)...")
    dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names["train"], num_proc=1)
    
    print("\nLoading Model with LoRA (PEFT)...")
    # Load model
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    # Enable gradient checkpointing to save VRAM
    model.config.use_cache = False
    model.generate = model.generate # Fix for PEFT warnings
    
    # Apply LoRA Config
    config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none"
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    
    # Training Arguments
    print("\nStarting Training...")
    print("This will take significantly less time than full training due to LoRA.")
    print("Estimated time: ~5-15 minutes depending on GPU.\n")
    
    training_args = Seq2SeqTrainingArguments(
        output_dir="robot-assistant/models/whisper-finetuned",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=1e-3,
        num_train_epochs=5,                       # 5 epochs for speed and efficiency
        fp16=True,                                # Use mixed precision on Ampere/RTX GPUs
        eval_strategy="steps",                   # Replaced evaluation_strategy with eval_strategy
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=50,
        eval_steps=50,
        logging_steps=10,
        report_to=["none"],                       # No wandb
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
    )
    
    # Data collator specially for Whisper
    import dataclasses
    from typing import Any, Dict, List, Union
    
    @dataclasses.dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any
        
        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
            
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
            
            # replace padding with -100 to ignore loss correctly
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            
            # if bos token is appended in previous tokenization step, cut it off
            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
                labels = labels[:, 1:]
                
            batch["labels"] = labels
            return batch
            
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    
    # WER metric
    metric = evaluate.load("wer")
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    # Initialize Trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )
    
    trainer.train()
    
    # Save the final model
    best_model_dir = "robot-assistant/models/whisper-finetuned-best"
    trainer.save_model(best_model_dir)
    processor.save_pretrained(best_model_dir)
    print(f"\nTraining Complete! Best model saved to {best_model_dir}")

if __name__ == "__main__":
    finetune_whisper()
