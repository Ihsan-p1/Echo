import cv2
import torch
import numpy as np
import threading
import queue
import time
from ultralytics import YOLO
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModelForSpeechSeq2Seq, # Add these for Whisper
    AutoProcessor
)
from peft import PeftModel
import sounddevice as sd
from colorama import Fore, Back, Style, init

# Initialize colorama
init(autoreset=True)
# import scipy.io.wavfile as wav # Removed as no longer needed

# --- CONFIGURATION ---
VISION_MODEL_PATH = "robot-assistant/models/yolo11n.pt" # Use base model for broad vision
WHISPER_MODEL_PATH = "robot-assistant/models/whisper-finetuned-best"
LLM_BASE_MODEL = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
LLM_ADAPTER_PATH = "robot-assistant/models/llm-finetuned-adapter"

class EchoRobot:
    def __init__(self):
        print("Initializing Echo Robot Systems...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. Load Vision (YOLO)
        print("Loading Vision Module (YOLO)...")
        self.vision_model = YOLO(VISION_MODEL_PATH)
        self.visual_context = []
        
        # 2. Load STT (Whisper)
        print("Loading Audio Module (Whisper)...")
        self.stt_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            WHISPER_MODEL_PATH,
            torch_dtype=torch.float16, # Use half for speed
            low_cpu_mem_usage=True,
            use_safetensors=True
        ).to(self.device)
        self.stt_processor = AutoProcessor.from_pretrained(WHISPER_MODEL_PATH)
        
        # 3. Load Brain (Llama 3.2 3B + LoRA)
        print("Loading Brain Module (Llama 3.2 3B LoRA)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            LLM_BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto"
        )
        self.llm_model = PeftModel.from_pretrained(base_model, LLM_ADAPTER_PATH)
        self.llm_tokenizer = AutoTokenizer.from_pretrained(LLM_BASE_MODEL)
        self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
        
        print("Robot Systems Online! Press 'q' to quit, 's' to speak.")

    def get_visual_context(self, frame):
        # Lower confidence threshold to detect more objects
        results = self.vision_model(frame, conf=0.15, verbose=False)
        counts = {}
        for r in results:
            for c in r.boxes.cls:
                name = self.vision_model.names[int(c)]
                counts[name] = counts.get(name, 0) + 1
        
        # Format as list of strings like ['3 persons', '1 laptop']
        context = [f"{count} {name}{'s' if count > 1 else ''}" for name, count in counts.items()]
        return context

    def record_audio(self):
        try:
            duration = 3  # seconds
            fs = 16000
            print("\n[LISTENING]... (Please speak now for 3 seconds)")
            recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
            sd.wait()
            print("[AUDIO CAPTURED] Transcribing directly (No ffmpeg/file)...")
            
            # Convert to float32 1D numpy array
            audio_data = recording.flatten().astype(np.float32)
            
            # Preprocess the audio
            input_features = self.stt_processor(
                audio_data, 
                sampling_rate=fs, 
                return_tensors="pt"
            ).input_features.to(self.device).to(torch.float16)

            # Generate transcription
            predicted_ids = self.stt_model.generate(
                input_features,
                language="en", # Explicitly set to English
                task="transcribe"
            )
            
            # Decode the ids
            transcription = self.stt_processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0]
            
            return transcription.strip()
        except Exception as e:
            print(f"\n[AUDIO ERROR] Failed to record/transcribe: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def query_brain(self, visual_context, user_query):
        try:
            print("[BRAIN] Analyzing visual context and query...")
            
            # Revert to exact prompt format used during fine-tuning for max stability
            instruction = "You are a friendly robot assistant. You can see through the camera. Analyze the visual context and respond to the user. Extract a motor command if needed (FORWARD, BACKWARD, LEFT, RIGHT, STOP, NONE)."
            input_text = f"Visual Context: {visual_context}\nUser Query: {user_query}"
            prompt = f"### Instruction: {instruction}\n### Input: {input_text}\n### Response: "
            
            inputs = self.llm_tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,
                    pad_token_id=self.llm_tokenizer.eos_token_id,
                    eos_token_id=self.llm_tokenizer.eos_token_id
                )
                
            full_response = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the newly generated part after ### Response:
            if "### Response: " in full_response:
                response = full_response.split("### Response: ")[-1].strip()
            else:
                response = full_response.strip()
                
            return response
        except Exception as e:
            print(f"\n[BRAIN ERROR] LLM Inference failed: {e}")
            return "<response>Sorry, my brain encountered an error.</response>\n<command>STOP</command>"

    def run(self):
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Continuous Vision
            self.visual_context = self.get_visual_context(frame)
            
            # Display current visual context on screen
            cv2.putText(frame, f"Context: {', '.join(self.visual_context)}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Echo Robot - Vision System', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                user_query = self.record_audio()
                if user_query:
                    print(f"\n{Fore.CYAN}{Style.BRIGHT}USER:{Style.RESET_ALL} {user_query}")
                    
                    # Brain processing
                    response_block = self.query_brain(self.visual_context, user_query)
                    
                    # Parse Response and Command for cleaner display
                    resp_text = response_block # Default to raw if no tags found
                    cmd_text = "NONE"
                    
                    if "<response>" in response_block and "</response>" in response_block:
                        resp_text = response_block.split("<response>")[1].split("</response>")[0]
                    elif "<response>" in response_block:
                         resp_text = response_block.split("<response>")[1].split("<")[0]
                         
                    if "<command>" in response_block and "</command>" in response_block:
                        cmd_text = response_block.split("<command>")[1].split("</command>")[0]
                    elif "<command>" in response_block:
                         cmd_text = response_block.split("<command>")[1].split("<")[0]

                    print(f"\n{Fore.YELLOW}{'='*50}")
                    print(f"{Fore.GREEN}{Style.BRIGHT}ROBOT BRAIN RESPONSE:{Style.RESET_ALL}")
                    print(f"{Fore.WHITE}{resp_text}")
                    print(f"{Fore.MAGENTA}{Style.BRIGHT}EXECUTING COMMAND:{Style.RESET_ALL} {Fore.RED}{cmd_text}")
                    print(f"{Fore.YELLOW}{'='*50}\n")
                    
                    # Log to file for tracking
                    with open("robot_history.log", "a") as f:
                        f.write(f"[{time.ctime()}] Context: {self.visual_context} | Query: {user_query}\nResult: {response_block}\n\n")

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    robot = EchoRobot()
    robot.run()
