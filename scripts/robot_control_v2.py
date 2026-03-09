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
WHISPER_MODEL_PATH = "robot-assistant/models/whisper-finetuned-best"
LLM_BASE_MODEL = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
LLM_ADAPTER_PATH = "robot-assistant/models/llm-finetuned-adapter"
JARVIS_VOICE_MODEL = "robot-assistant/voices/jarvis/en/en_GB/jarvis/high/jarvis-high.onnx"
JARVIS_VOICE_CONFIG = "robot-assistant/voices/jarvis/en/en_GB/jarvis/high/jarvis-high.onnx.json"
HAND_LANDMARKER_PATH = "robot-assistant/models/hand_landmarker.task"
YOLO_CONF_THRESHOLD = 0.45  # c4: raised from 0.15 to reduce false positives

# MediaPipe Hand Connections for drawing (since mp.solutions.drawing_utils is removed)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),       # thumb
    (0,5),(5,6),(6,7),(7,8),       # index
    (0,9),(9,10),(10,11),(11,12),  # middle
    (0,13),(13,14),(14,15),(15,16),# ring
    (0,17),(17,18),(18,19),(19,20),# pinky
    (5,9),(9,13),(13,17),          # palm
]

# Gesture-to-command mapping (deterministic, bypass LLM)
GESTURE_COMMAND_MAP = {
    "pointing":    "FORWARD",
    "open_palm":   "STOP",
    "fist":        "BACKWARD",
    "thumb_left":  "LEFT",
    "thumb_right": "RIGHT",
}

GESTURE_RESPONSE_MAP = {
    "pointing":    "Moving forward.",
    "open_palm":   "Stopping.",
    "fist":        "Moving backward.",
    "thumb_left":  "Turning left.",
    "thumb_right": "Turning right.",
}

class GestureDetector:
    """Background gesture detector using MediaPipe HandLandmarker (tasks API)."""
    GESTURE_MAP = {
        "open_palm":   "STOP",
        "pointing":    "FORWARD",
        "fist":        "BACKWARD",
        "thumb_left":  "LEFT",
        "thumb_right": "RIGHT",
        "thumbs_up":   "NONE",
        "none":        "NONE",
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

        # Thumb: distance-based (more stable than x comparison)
        wrist = landmarks[0]
        thumb_tip = landmarks[4]
        thumb_pip = landmarks[3]
        thumb_dist_tip = abs(thumb_tip.x - wrist.x)
        thumb_dist_pip = abs(thumb_pip.x - wrist.x)
        thumb_up = thumb_dist_tip > thumb_dist_pip

        # Other fingers: tip.y < pip.y = extended
        index_up  = landmarks[tip_ids[1]].y < landmarks[pip_ids[1]].y
        middle_up = landmarks[tip_ids[2]].y < landmarks[pip_ids[2]].y
        ring_up   = landmarks[tip_ids[3]].y < landmarks[pip_ids[3]].y
        pinky_up  = landmarks[tip_ids[4]].y < landmarks[pip_ids[4]].y

        # Open palm: all 4 fingers up → STOP
        if index_up and middle_up and ring_up and pinky_up:
            return "open_palm"

        # Pointing: only index finger up → FORWARD
        if index_up and not middle_up and not ring_up and not pinky_up:
            return "pointing"

        # Fist: all fingers closed, thumb not extended → BACKWARD
        if not index_up and not middle_up and not ring_up and not pinky_up and not thumb_up:
            return "fist"

        # Thumb only extended → check direction for LEFT/RIGHT/thumbs_up
        if thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
            if thumb_tip.x < wrist.x - 0.12:
                return "thumb_left"
            elif thumb_tip.x > wrist.x + 0.12:
                return "thumb_right"
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

    def close(self):
        """Explicitly close MediaPipe landmarker to prevent TypeError on exit."""
        try:
            self.landmarker.close()
        except Exception:
            pass


class WakeWordListener:
    """Background wake word detector using openwakeword."""
    def __init__(self, sensitivity=0.5):
        self.oww = WakeWordModel(
            wakeword_models=["hey_jarvis"],
            inference_framework="onnx"
        )
        self.sensitivity = sensitivity
        self._activated = threading.Event()
        self._running = False
        self._thread = None

    def start(self):
        """Start listening for wake word in background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _listen_loop(self):
        """Continuously listen for wake word."""
        pa = pyaudio.PyAudio()
        mic = pa.open(
            rate=16000, channels=1,
            format=pyaudio.paInt16,
            input=True, frames_per_buffer=1280
        )

        while self._running:
            chunk = np.frombuffer(mic.read(1280, exception_on_overflow=False), dtype=np.int16)
            preds = self.oww.predict(chunk)
            for model_name, score in preds.items():
                if score > self.sensitivity:
                    print(f"\n{Fore.GREEN}{Style.BRIGHT}[WAKE]{Style.RESET_ALL} '{model_name}' detected! (score: {score:.2f})")
                    self._activated.set()
                    self.oww.reset()  # prevent rapid re-trigger
                    time.sleep(0.3)

        mic.stop_stream()
        mic.close()
        pa.terminate()

    def check_activated(self):
        """Check if wake word was detected (non-blocking). Clears the flag."""
        if self._activated.is_set():
            self._activated.clear()
            return True
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
            dtype=torch.float16, # Use half for speed
            low_cpu_mem_usage=True,
            use_safetensors=True
        ).to(self.device)
        self.stt_processor = AutoProcessor.from_pretrained(WHISPER_MODEL_PATH)
        
        # 3. Load Brain (Llama 3.2 3B + LoRA)
        print("Loading Brain Module (Llama 3.2 3B LoRA)...")
        base_model = AutoModelForCausalLM.from_pretrained(
            LLM_BASE_MODEL,
            device_map="auto"
            # quantization_config not needed — unsloth model bundles its own config
        )
        self.llm_model = PeftModel.from_pretrained(base_model, LLM_ADAPTER_PATH)
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
            
            instruction = """You are Echo, a friendly robot assistant with a camera.
Always respond using this exact XML format:

<response>Your spoken response here. Natural, 1-2 sentences max.</response>
<command>FORWARD|BACKWARD|LEFT|RIGHT|STOP|NONE</command>

Rules:
- <response> is what you SAY OUT LOUD. Keep it conversational.
- <command> is the motor action. Use NONE if no movement needed.
- NEVER put command keywords inside <response>.
- NEVER output any text outside the XML tags.
- Always respond in English.

Example:
Visual: person (center, close), laptop (left, nearby)
User: What do you see?
<response>I can see a person right in front of me and a laptop to my left.</response>
<command>NONE</command>"""
            input_text = f"Visual Context: {visual_context}\nUser Input: {user_input}"
            prompt = f"### Instruction: {instruction}\n### Input: {input_text}\n### Response: "
            
            inputs = self.llm_tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.3,
                    top_p=0.9,
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
        """Extract clean speech text from LLM output, stripping XML and command artifacts."""
        # Priority: extract content from <response> tag if present
        match = re.search(r'<response>(.*?)</response>', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Fallback: strip command tags and their content
        text = re.sub(r'<command>.*?</command>', '', text, flags=re.DOTALL)
        # Strip remaining XML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove lines that are solely a command keyword
        lines = text.split('\n')
        clean = [l.strip() for l in lines
                 if l.strip()
                 and not re.match(
                     r'^(FORWARD|BACKWARD|LEFT|RIGHT|STOP|NONE)$',
                     l.strip(), re.I
                 )]
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
                    sd.play(audio_data, samplerate=self.tts_voice.config.sample_rate)
                    sd.wait()
            except Exception as e:
                print(f"[TTS ERROR] Speech failed: {e}")

        threading.Thread(target=_speak, daemon=True).start()

    def handle_gesture_command(self, gesture):
        """Gesture command — bypass LLM completely. Deterministic, <1ms."""
        command = GESTURE_COMMAND_MAP.get(gesture)
        response = GESTURE_RESPONSE_MAP.get(gesture, "Gesture not recognized.")

        if command:
            self.send_motor_command(command)
            self.speak_async(response)
            print(f"\n{Fore.YELLOW}{'='*50}")
            print(f"{Fore.YELLOW}{Style.BRIGHT}[GESTURE]{Style.RESET_ALL} {gesture} \u2192 {Fore.RED}{command}")
            print(f"{Fore.YELLOW}{'='*50}\n")
            # Log gesture command
            self.logger.info(
                f"Intent: GESTURE_COMMAND | Source: gesture | Context: {self.visual_context} | "
                f"Input: gesture:{gesture} | Response: {response} | Command: {command}")
        else:
            print(f"{Fore.YELLOW}[GESTURE]{Style.RESET_ALL} '{gesture}' \u2014 no mapping found.")

    def send_motor_command(self, command):
        """Placeholder for laptop testing. Replace body when integrating ESP32."""
        # TODO: self.serial.write(f"MOTOR:{command}\n".encode()) when ESP32 connected
        print(f"{Fore.MAGENTA}[MOTOR]{Style.RESET_ALL} >> {command} (simulated \u2014 ESP32 not connected)")

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
        cap = cv2.VideoCapture(0)

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

            cv2.imshow('Echo Robot - Vision System', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and self.robot_state == "IDLE":
                # Manual voice trigger (fallback)
                self.audio_trigger.put("manual")
            elif key == ord('g'):
                # Gesture trigger — bypass LLM, deterministic mapping
                current_gesture = self.gesture_detector.get_gesture()
                if current_gesture != "none" and self.robot_state == "IDLE":
                    self.handle_gesture_command(current_gesture)
                else:
                    print(f"{Fore.YELLOW}[GESTURE]{Style.RESET_ALL} No gesture detected.")

        # Cleanup
        self.wakeword.stop()
        self.audio_trigger.put(None)  # shutdown worker
        self.gesture_detector.close()  # explicit close — prevent TypeError on exit
        cap.release()
        cv2.destroyAllWindows()

    def _set_idle(self):
        self.robot_state = "IDLE"

if __name__ == "__main__":
    robot = EchoRobot()
    robot.run()
