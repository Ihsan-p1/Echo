import cv2
import torch
import numpy as np
import threading
import queue
import time
import wave
import io
import re
import logging
from logging.handlers import RotatingFileHandler
import webrtcvad
import mediapipe as mp
import pyaudio
from openwakeword.model import Model as WakeWordModel
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
from piper.voice import PiperVoice
from colorama import Fore, Back, Style, init

# Initialize colorama
init(autoreset=True)
# import scipy.io.wavfile as wav # Removed as no longer needed

# --- CONFIGURATION ---
VISION_MODEL_PATH = "robot-assistant/models/yolo11n.pt"
# WHISPER_MODEL_PATH = "robot-assistant/models/whisper-finetuned-best"
WHISPER_MODEL_PATH = "openai/whisper-small"
LLM_BASE_MODEL = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
LLM_ADAPTER_PATH = "robot-assistant/models/llm-finetuned-adapter"
# JARVIS_VOICE_MODEL = "robot-assistant/voices/jarvis/en/en_GB/jarvis/high/jarvis-high.onnx"
# JARVIS_VOICE_CONFIG = "robot-assistant/voices/jarvis/en/en_GB/jarvis/high/jarvis-high.onnx.json"
JARVIS_VOICE_MODEL = "robot-assistant/voices/en_US-lessac-medium.onnx"
JARVIS_VOICE_CONFIG = "robot-assistant/voices/en_US-lessac-medium.onnx.json"
HAND_LANDMARKER_PATH = "robot-assistant/models/hand_landmarker.task"
YOLO_CONF_THRESHOLD = 0.45  # c4: raised from 0.15 to reduce false positives

# --
# MediaPipe Hand Connections for drawing (since mp.solutions.drawing_utils is removed)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),       # thumb
    (0,5),(5,6),(6,7),(7,8),       # index
    (0,9),(9,10),(10,11),(11,12),  # middle
    (0,13),(13,14),(14,15),(15,16),# ring
    (0,17),(17,18),(18,19),(19,20),# pinky
    (5,9),(9,13),(13,17),          # palm
]

class GestureDetector:
    """Background gesture detector using MediaPipe HandLandmarker (tasks API)."""
    GESTURE_MAP = {
        "open_palm": "STOP",
        "pointing": "FORWARD",
        "thumbs_up": "NONE",
        "none": "NONE"
    }

    def __init__(self):
        from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
        from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
        from mediapipe.tasks.python import BaseOptions

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=HAND_LANDMARKER_PATH),
            running_mode=VisionTaskRunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.75,
            min_tracking_confidence=0.5,
        )
        self.landmarker = HandLandmarker.create_from_options(options)
        self.current_gesture = "none"
        self._lock = threading.Lock()
        self._frame_ts = 0

    def _classify_gesture(self, landmarks):
        """Classify hand gesture from NormalizedLandmark list."""
        tip_ids = [4, 8, 12, 16, 20]
        pip_ids = [3, 6, 10, 14, 18]

        fingers_up = []

        # Thumb: compare x
        if landmarks[tip_ids[0]].x < landmarks[pip_ids[0]].x:
            fingers_up.append(True)
        else:
            fingers_up.append(False)

        # Other fingers: compare y (tip above pip = extended)
        for i in range(1, 5):
            if landmarks[tip_ids[i]].y < landmarks[pip_ids[i]].y:
                fingers_up.append(True)
            else:
                fingers_up.append(False)

        if all(fingers_up):
            return "open_palm"
        elif fingers_up[1] and not any(fingers_up[2:]):
            return "pointing"
        elif fingers_up[0] and not any(fingers_up[1:]):
            return "thumbs_up"
        return "none"

    def _draw_landmarks(self, frame, landmarks, h, w):
        """Draw hand landmarks and connections on the frame."""
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, pts[start], pts[end], (0, 255, 200), 2)
        for pt in pts:
            cv2.circle(frame, pt, 4, (0, 200, 255), -1)

    def process_frame(self, frame):
        """Process a frame for gesture detection. Call from main loop."""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._frame_ts += 33  # ~30fps in ms
        result = self.landmarker.detect_for_video(mp_image, self._frame_ts)

        gesture = "none"
        if result.hand_landmarks:
            for hand_lm in result.hand_landmarks:
                gesture = self._classify_gesture(hand_lm)
                self._draw_landmarks(frame, hand_lm, h, w)

        with self._lock:
            self.current_gesture = gesture
        return gesture

    def get_gesture(self):
        """Thread-safe getter for current gesture."""
        with self._lock:
            return self.current_gesture

    def get_command(self):
        """Map current gesture to motor command."""
        return self.GESTURE_MAP.get(self.get_gesture(), "NONE")


class WakeWordListener:
    """Background wake word detector (Bypassed for Codespaces)."""
    def __init__(self, sensitivity=0.5):
        print("Wake Word Module Bypassed (No Microphone in Cloud VM)")
        self.sensitivity = sensitivity
        self._activated = threading.Event()
        self._running = False
        self._thread = None

    def start(self):
        print("Audio listening disabled. Robot is deaf in this simulation.")

    def stop(self):
        pass
        
    def wait_for_wakeword(self):
        import time
        time.sleep(1)
        return False

    def check_activated(self):
        # Always return False in simulation since there is no microphone
        # We rely on keyboard input ('s' or 'g') instead.
        return False

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
        # self.llm_model = PeftModel.from_pretrained(base_model, LLM_ADAPTER_PATH)
        # self.llm_model = PeftModel.from_pretrained(base_model, LLM_ADAPTER_PATH)
        self.llm_model = base_model # Bypass pemuatan LoRA adapter
        self.llm_tokenizer = AutoTokenizer.from_pretrained(LLM_BASE_MODEL)
        self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
        
        # 4. Load Gesture Detector (MediaPipe)
        print("Loading Gesture Module (MediaPipe)...")
        self.gesture_detector = GestureDetector()
        
        # 5. Setup Wake Word Listener
        print("Loading Wake Word Module (openwakeword)...")
        self.wakeword = WakeWordListener(sensitivity=0.5)
        
        # 6. Load TTS Voice once (c1: avoid reloading .onnx every speak call)
        print("Loading TTS Voice (Piper)...")
        self.tts_voice = PiperVoice.load(JARVIS_VOICE_MODEL, config_path=JARVIS_VOICE_CONFIG)
        
        # 7. Threading infrastructure
        self.audio_trigger = queue.Queue()    # signals to start audio pipeline
        self.response_queue = queue.Queue()   # results from audio pipeline
        self.robot_state = "IDLE"             # IDLE, LISTENING, THINKING, SPEAKING
        
        # 8. Setup rotating logger (c6: 5MB max, 3 backups)
        log_handler = RotatingFileHandler(
            'robot_history.log', maxBytes=5*1024*1024, backupCount=3)
        logging.basicConfig(
            handlers=[log_handler], level=logging.INFO,
            format='%(asctime)s | %(message)s')
        self.logger = logging.getLogger('echo')
        
        # c2: VRAM monitoring after all models loaded
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            pct = alloc / total * 100
            color = Fore.RED if pct > 90 else Fore.YELLOW if pct > 75 else Fore.GREEN
            print(f"{color}[VRAM] {alloc:.2f}GB / {total:.2f}GB ({pct:.0f}% used){Style.RESET_ALL}")
            if pct > 90:
                print(f"{Fore.RED}{Style.BRIGHT}[VRAM WARNING] >90% used — high OOM risk!{Style.RESET_ALL}")
        
        print(f"{Fore.GREEN}{Style.BRIGHT}Robot Systems Online!{Style.RESET_ALL}")
        print(f"  Say {Fore.CYAN}'Hey Jarvis'{Style.RESET_ALL} or press {Fore.CYAN}'s'{Style.RESET_ALL} to speak")
        print(f"  Press {Fore.YELLOW}'g'{Style.RESET_ALL} for gesture, {Fore.RED}'q'{Style.RESET_ALL} to quit")

    def get_visual_context(self, frame):
        """Extract spatial-aware visual context with position, distance proxy, and confidence."""
        h, w = frame.shape[:2]
        results = self.vision_model(frame, conf=YOLO_CONF_THRESHOLD, verbose=False)
        context_items = []

        for r in results:
            for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                name = self.vision_model.names[int(cls)]
                x1, y1, x2, y2 = box.tolist()
                cx = (x1 + x2) / 2

                # Horizontal position
                if cx < w * 0.33:
                    h_pos = "left"
                elif cx < w * 0.66:
                    h_pos = "center"
                else:
                    h_pos = "right"

                # Distance proxy via bbox area ratio
                area_ratio = ((x2 - x1) * (y2 - y1)) / (w * h)
                if area_ratio > 0.3:
                    dist = "close"
                elif area_ratio > 0.1:
                    dist = "nearby"
                else:
                    dist = "far"

                # Draw bounding box on frame — color by distance
                color_map = {"close": (0, 0, 255), "nearby": (0, 255, 255), "far": (0, 255, 0)}
                color = color_map.get(dist, (255, 255, 255))
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = f"{name} {conf:.0%}"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (int(x1), int(y1) - lh - 6), (int(x1) + lw, int(y1)), color, -1)
                cv2.putText(frame, label, (int(x1), int(y1) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                context_items.append(
                    f"{name} ({h_pos}, {dist}, conf:{conf:.2f})"
                )

        return context_items
        # Output example: ["person (center, close, conf:0.91)", "laptop (left, nearby, conf:0.87)"]

    def record_with_vad(self, sample_rate=16000, silence_threshold=1.5, max_duration=8):
        """Record audio with Voice Activity Detection — auto-stops after speech ends."""
        vad = webrtcvad.Vad(2)  # aggressiveness 0-3 (2 = balanced)
        chunk_duration = 0.03   # 30ms chunks (required by webrtcvad)
        chunk_size = int(sample_rate * chunk_duration)

        frames = []
        silence_chunks = 0
        max_silence = int(silence_threshold / chunk_duration)
        max_chunks = int(max_duration / chunk_duration)
        speaking = False

        print(f"\n{Fore.CYAN}[LISTENING]{Style.RESET_ALL} Speak now... (auto-stops after {silence_threshold}s silence, max {max_duration}s)")

        try:
            with sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16') as stream:
                while len(frames) < max_chunks:
                    chunk, _ = stream.read(chunk_size)
                    chunk_bytes = chunk.tobytes()

                    is_speech = vad.is_speech(chunk_bytes, sample_rate)

                    if is_speech:
                        speaking = True
                        silence_chunks = 0
                        frames.append(chunk)
                    elif speaking:
                        silence_chunks += 1
                        frames.append(chunk)  # keep trailing silence for natural cutoff
                        if silence_chunks > max_silence:
                            print(f"{Fore.GREEN}[VAD]{Style.RESET_ALL} Speech ended ({len(frames) * chunk_duration:.1f}s captured)")
                            break
        except Exception as e:
            print(f"{Fore.RED}[VAD ERROR]{Style.RESET_ALL} Recording failed: {e}")
            return np.array([], dtype=np.float32)

        if not frames:
            print(f"{Fore.YELLOW}[VAD]{Style.RESET_ALL} No speech detected.")
            return np.array([], dtype=np.float32)

        audio = np.concatenate(frames).flatten().astype(np.float32)
        audio = audio / 32768.0  # normalize int16 → float32
        return audio

    def record_audio(self):
        """Record via VAD, then transcribe with Whisper."""
        try:
            audio_data = self.record_with_vad()

            if audio_data.size == 0:
                return ""

            print("[AUDIO CAPTURED] Transcribing...")

            # Preprocess the audio
            input_features = self.stt_processor(
                audio_data,
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features.to(self.device).to(torch.float16)

            # Generate transcription
            predicted_ids = self.stt_model.generate(
                input_features,
                language="en",
                task="transcribe"
            )

            # Decode the ids
            transcription = self.stt_processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )[0]

            return transcription.strip()
        except Exception as e:
            print(f"\n{Fore.RED}[AUDIO ERROR]{Style.RESET_ALL} Failed to record/transcribe: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def query_brain(self, visual_context, user_input):
        """Send visual context + user input (voice and/or gesture) to LLM with unified intent system."""
        try:
            print("[BRAIN] Analyzing visual context and query...")
            
            instruction = """You are Echo, a friendly robot assistant with camera and motor control.
Analyze the visual context and user input, then respond with structured XML:

<intent>QUERY|COMMAND|GESTURE_COMMAND</intent>
<response>Your natural language response here</response>
<command>FORWARD|BACKWARD|LEFT|RIGHT|STOP|NONE</command>

Intent types:
- QUERY: user asks about what robot sees or general questions
- COMMAND: user gives movement instruction via speech
- GESTURE_COMMAND: movement from detected hand gesture

Rules:
- Always respond in English
- Keep responses to 1-2 sentences
- If gesture is detected AND voice is empty, use gesture as the command source
- If both gesture and voice are present, voice takes priority"""
            input_text = f"Visual Context: {visual_context}\nUser Input: {user_input}"
            prompt = f"### Instruction: {instruction}\n### Input: {input_text}\n### Response: "
            
            inputs = self.llm_tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    **inputs,
                    max_new_tokens=150,
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
        except torch.cuda.OutOfMemoryError:
            # c2 + c5: Handle VRAM overflow gracefully
            torch.cuda.empty_cache()
            print(f"\n{Fore.RED}{Style.BRIGHT}[OOM]{Style.RESET_ALL} VRAM full — cleared cache")
            self.speak_async("Memory is full, please try again.")
            return None
        except Exception as e:
            # c5: Verbal error feedback instead of silent failure
            print(f"\n{Fore.RED}[BRAIN ERROR]{Style.RESET_ALL} LLM Inference failed: {e}")
            self.speak_async("Sorry, I could not process that.")
            return None

    def parse_llm_output(self, raw_output):
        """Parse structured XML tags from LLM output."""
        def extract_tag(text, tag):
            match = re.search(f'<{tag}>(.*?)</{tag}>', text, re.DOTALL)
            return match.group(1).strip() if match else None

        intent = extract_tag(raw_output, "intent") or "QUERY"
        response = extract_tag(raw_output, "response") or raw_output
        command = extract_tag(raw_output, "command") or "NONE"
        
        # Validate command
        valid_commands = {"FORWARD", "BACKWARD", "LEFT", "RIGHT", "STOP", "NONE"}
        if command.upper() not in valid_commands:
            command = "NONE"
        else:
            command = command.upper()
            
        return intent, response, command

    def _clean_speech_text(self, text):
        """Remove command/metadata artifacts from text before speaking (c3: less aggressive)."""
        text = re.sub(r'<[^>]+>', '', text)  # strip XML tags
        lines = text.split('\n')
        clean = []
        for line in lines:
            s = line.strip()
            if not s:
                continue
            if s.startswith('###'):
                continue
            if re.match(r'^(Motor Command|Command)\s*:', s, re.I):
                continue
            # c3: Only remove lines that are SOLELY a command keyword
            if re.match(r'^(FORWARD|BACKWARD|LEFT|RIGHT|STOP|NONE)$', s, re.I):
                continue
            clean.append(s)
        return re.sub(r'\s+', ' ', ' '.join(clean)).strip()

    def speak_async(self, text):
        clean_text = self._clean_speech_text(text)
        if not clean_text:
            return
            
        def _speak():
            try:
                print(f"{Fore.BLUE}{Style.BRIGHT}[\U0001f50a SPEAKING]{Style.RESET_ALL} {clean_text}")
                # c1: Use pre-loaded TTS voice instead of reloading every call
                audio_bytes = b''
                for audio_chunk in self.tts_voice.synthesize(clean_text):
                    audio_bytes += audio_chunk.audio_int16_bytes
                
                if audio_bytes:
                    audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
                    # sd.play(audio_data, samplerate=self.tts_voice.config.sample_rate)
                    # sd.wait()
            except Exception as e:
                print(f"[TTS ERROR] Speech failed: {e}")

        threading.Thread(target=_speak, daemon=True).start()



    def _process_brain_response(self, user_input, source="voice"):
        """Unified brain processing pipeline: LLM → parse → display → speak → log."""
        raw_output = self.query_brain(self.visual_context, user_input)
        
        # c5: Handle None from query_brain (error already spoken)
        if raw_output is None:
            return None, None, "NONE"
        
        intent, resp_text, cmd_text = self.parse_llm_output(raw_output)

        # Display
        intent_colors = {"QUERY": Fore.CYAN, "COMMAND": Fore.MAGENTA, "GESTURE_COMMAND": Fore.YELLOW}
        intent_color = intent_colors.get(intent, Fore.WHITE)

        print(f"\n{Fore.YELLOW}{'='*50}")
        print(f"{intent_color}{Style.BRIGHT}[{intent}]{Style.RESET_ALL} via {source}")
        print(f"{Fore.GREEN}{Style.BRIGHT}RESPONSE:{Style.RESET_ALL} {resp_text}")
        print(f"{Fore.MAGENTA}{Style.BRIGHT}COMMAND:{Style.RESET_ALL} {Fore.RED}{cmd_text}")
        print(f"{Fore.YELLOW}{'='*50}\n")

        # Speak the response
        if resp_text and resp_text != "N/A":
            self.speak_async(resp_text)

        # c6: Use rotating logger instead of raw file append
        self.logger.info(
            f"Intent: {intent} | Source: {source} | Context: {self.visual_context} | "
            f"Input: {user_input} | Response: {resp_text} | Command: {cmd_text}")

        return intent, resp_text, cmd_text

    def _audio_pipeline_worker(self):
        """Worker thread: wake word / manual trigger → VAD → STT → LLM → queue response."""
        while True:
            trigger_source = self.audio_trigger.get()  # blocks until trigger
            if trigger_source is None:
                break  # shutdown signal

            try:
                # Record with VAD
                self.robot_state = "LISTENING"
                audio_data = self.record_with_vad()

                if audio_data.size == 0:
                    self.robot_state = "IDLE"
                    continue

                # Transcribe
                self.robot_state = "THINKING"
                print("[AUDIO CAPTURED] Transcribing...")
                input_features = self.stt_processor(
                    audio_data, sampling_rate=16000, return_tensors="pt"
                ).input_features.to(self.device).to(torch.float16)

                predicted_ids = self.stt_model.generate(
                    input_features, language="en", task="transcribe"
                )
                transcript = self.stt_processor.batch_decode(
                    predicted_ids, skip_special_tokens=True
                )[0].strip()

                if not transcript:
                    self.robot_state = "IDLE"
                    continue

                print(f"\n{Fore.CYAN}{Style.BRIGHT}USER ({trigger_source}):{Style.RESET_ALL} {transcript}")

                # Combine with gesture
                current_gesture = self.gesture_detector.get_gesture()
                user_input = f"voice: {transcript}"
                if current_gesture != "none":
                    user_input += f" | gesture: {current_gesture}"

                # LLM
                raw_output = self.query_brain(self.visual_context, user_input)
                
                # c5: Handle None from query_brain
                if raw_output is None:
                    self.robot_state = "IDLE"
                    continue
                
                intent, resp_text, cmd_text = self.parse_llm_output(raw_output)

                # Queue result for main thread display
                self.response_queue.put((intent, resp_text, cmd_text, user_input, trigger_source))

            except Exception as e:
                print(f"{Fore.RED}[PIPELINE ERROR]{Style.RESET_ALL} {e}")
                self.robot_state = "IDLE"

    def run(self):
        # cap = cv2.VideoCapture(0)
        cap = cv2.VideoCapture('sample_video.mp4')

        # Start background threads
        self.wakeword.start()
        audio_worker = threading.Thread(target=self._audio_pipeline_worker, daemon=True)
        audio_worker.start()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # --- Continuous Vision (YOLO) — never blocks ---
            self.visual_context = self.get_visual_context(frame)

            # --- Continuous Gesture Detection (MediaPipe) ---
            gesture = self.gesture_detector.process_frame(frame)

            # --- Check wake word activation (non-blocking) ---
            if self.wakeword.check_activated() and self.robot_state == "IDLE":
                self.audio_trigger.put("wake_word")

            # --- Check response queue (non-blocking) ---
            try:
                intent, resp_text, cmd_text, user_input, source = self.response_queue.get_nowait()

                # Display
                intent_colors = {"QUERY": Fore.CYAN, "COMMAND": Fore.MAGENTA, "GESTURE_COMMAND": Fore.YELLOW}
                intent_color = intent_colors.get(intent, Fore.WHITE)

                print(f"\n{Fore.YELLOW}{'='*50}")
                print(f"{intent_color}{Style.BRIGHT}[{intent}]{Style.RESET_ALL} via {source}")
                print(f"{Fore.GREEN}{Style.BRIGHT}RESPONSE:{Style.RESET_ALL} {resp_text}")
                print(f"{Fore.MAGENTA}{Style.BRIGHT}COMMAND:{Style.RESET_ALL} {Fore.RED}{cmd_text}")
                print(f"{Fore.YELLOW}{'='*50}\n")

                # Speak
                self.robot_state = "SPEAKING"
                if resp_text and resp_text != "N/A":
                    self.speak_async(resp_text)

                # c6: Use rotating logger
                self.logger.info(
                    f"Intent: {intent} | Source: {source} | Context: {self.visual_context} | "
                    f"Input: {user_input} | Response: {resp_text} | Command: {cmd_text}")

                # Return to IDLE after a short delay (let TTS start)
                threading.Timer(1.0, self._set_idle).start()

            except queue.Empty:
                pass

            # --- HUD ---
            state_colors = {
                "IDLE": (128, 128, 128), "LISTENING": (0, 255, 255),
                "THINKING": (255, 165, 0), "SPEAKING": (0, 200, 255)
            }
            state_color = state_colors.get(self.robot_state, (255, 255, 255))

            cv2.putText(frame, f"Context: {', '.join(self.visual_context)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            gesture_color = (0, 255, 255) if gesture != "none" else (128, 128, 128)
            cv2.putText(frame, f"Gesture: {gesture}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, gesture_color, 1)
            cv2.putText(frame, f"State: {self.robot_state}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)

            # cv2.imshow('Echo Robot - Vision System', frame)

            import sys, select
            # Mengecek input terminal tanpa menghentikan proses video YOLO di latar belakang
            if select.select([sys.stdin], [], [], 0.0)[0]:
                key_char = sys.stdin.readline().strip()
                if key_char == 'q':
                    break
                elif key_char == 's' and self.robot_state == "IDLE":
                    # bypass mikrofon, langsung ketik prompt di terminal
                    simulated_text = input(f"\n{Fore.CYAN}[MIC SIMULATOR] Ketik suara kamu: {Style.RESET_ALL}")
                    if simulated_text:
                        self._process_brain_response(f"voice: {simulated_text}", source="terminal")
                elif key_char == 'g':
                    current_gesture = self.gesture_detector.get_gesture()
                    if current_gesture != "none" and self.robot_state == "IDLE":
                        print(f"\n{Fore.YELLOW}{Style.BRIGHT}USER (gesture):{Style.RESET_ALL} {current_gesture}")
                        self._process_brain_response(f"gesture: {current_gesture}", source="gesture")
                    elif current_gesture == "none":
                        print(f"{Fore.YELLOW}[GESTURE]{Style.RESET_ALL} No gesture detected in video.")

        # Cleanup
        self.wakeword.stop()
        self.audio_trigger.put(None)  # shutdown worker
        cap.release()
        cv2.destroyAllWindows()

    def _set_idle(self):
        self.robot_state = "IDLE"

if __name__ == "__main__":
    robot = EchoRobot()
    robot.run()
