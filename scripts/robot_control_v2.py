import cv2
import torch
import numpy as np
import threading
import queue
import time
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
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
)
import sounddevice as sd
from piper.voice import PiperVoice
from colorama import Fore, Style, init

init(autoreset=True)

# --- CONFIGURATION ---
VISION_MODEL_PATH     = "robot-assistant/models/yolo11n.pt"
WHISPER_MODEL_PATH    = "robot-assistant/models/whisper-finetuned-best"
# FIX 1: Use base instruction-tuned model directly — LoRA adapter removed.
# Rationale: 100-step fine-tune on a small dataset degrades instruction-following
# of the base model rather than improving it. The base Llama-3.2-3B-Instruct is
# more reliably aligned with our strict XML prompt format.
LLM_BASE_MODEL        = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
JARVIS_VOICE_MODEL    = "robot-assistant/voices/jarvis/en/en_GB/jarvis/high/jarvis-high.onnx"
JARVIS_VOICE_CONFIG   = "robot-assistant/voices/jarvis/en/en_GB/jarvis/high/jarvis-high.onnx.json"
HAND_LANDMARKER_PATH  = "robot-assistant/models/hand_landmarker.task"
YOLO_CONF_THRESHOLD   = 0.45

VALID_COMMANDS = {"FORWARD", "BACKWARD", "LEFT", "RIGHT", "STOP", "NONE"}

# Explicit movement keywords — used by the safety gate before sending to LLM
# Any voice input that does NOT contain these words should never produce a movement command.
MOVEMENT_TRIGGER_WORDS = {
    "forward", "backward", "back", "left", "right", "turn", "move",
    "go", "stop", "halt", "reverse", "advance", "retreat",
}

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

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


# ---------------------------------------------------------------------------
# GestureDetector
# ---------------------------------------------------------------------------
class GestureDetector:
    """Deterministic gesture → command mapping via MediaPipe HandLandmarker."""

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
        tip_ids = [4, 8, 12, 16, 20]
        pip_ids = [3, 6, 10, 14, 18]

        wrist     = landmarks[0]
        thumb_tip = landmarks[4]
        thumb_pip = landmarks[3]
        thumb_up  = abs(thumb_tip.x - wrist.x) > abs(thumb_pip.x - wrist.x)

        index_up  = landmarks[tip_ids[1]].y < landmarks[pip_ids[1]].y
        middle_up = landmarks[tip_ids[2]].y < landmarks[pip_ids[2]].y
        ring_up   = landmarks[tip_ids[3]].y < landmarks[pip_ids[3]].y
        pinky_up  = landmarks[tip_ids[4]].y < landmarks[pip_ids[4]].y

        if index_up and middle_up and ring_up and pinky_up:
            return "open_palm"
        if index_up and not middle_up and not ring_up and not pinky_up:
            return "pointing"
        if not index_up and not middle_up and not ring_up and not pinky_up and not thumb_up:
            return "fist"
        if thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
            if thumb_tip.x < wrist.x - 0.12:
                return "thumb_left"
            elif thumb_tip.x > wrist.x + 0.12:
                return "thumb_right"
            return "thumbs_up"
        return "none"

    def _draw_landmarks(self, frame, landmarks, h, w):
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, pts[start], pts[end], (0, 255, 200), 2)
        for pt in pts:
            cv2.circle(frame, pt, 4, (0, 200, 255), -1)

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._frame_ts += 33
        result   = self.landmarker.detect_for_video(mp_image, self._frame_ts)

        gesture = "none"
        if result.hand_landmarks:
            for hand_lm in result.hand_landmarks:
                gesture = self._classify_gesture(hand_lm)
                self._draw_landmarks(frame, hand_lm, h, w)

        with self._lock:
            self.current_gesture = gesture
        return gesture

    def get_gesture(self):
        with self._lock:
            return self.current_gesture

    def get_command(self):
        return self.GESTURE_MAP.get(self.get_gesture(), "NONE")

    def close(self):
        try:
            self.landmarker.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# WakeWordListener
# ---------------------------------------------------------------------------
class WakeWordListener:
    def __init__(self, sensitivity=0.5):
        self.oww = WakeWordModel(
            wakeword_models=["hey_jarvis"],
            inference_framework="onnx"
        )
        self.sensitivity  = sensitivity
        self._activated   = threading.Event()
        self._running     = False
        self._thread      = None

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _listen_loop(self):
        pa  = pyaudio.PyAudio()
        mic = pa.open(rate=16000, channels=1, format=pyaudio.paInt16,
                      input=True, frames_per_buffer=1280)
        while self._running:
            chunk = np.frombuffer(mic.read(1280, exception_on_overflow=False), dtype=np.int16)
            preds = self.oww.predict(chunk)
            for model_name, score in preds.items():
                if score > self.sensitivity:
                    print(f"\n{Fore.GREEN}{Style.BRIGHT}[WAKE]{Style.RESET_ALL} "
                          f"'{model_name}' detected! (score: {score:.2f})")
                    self._activated.set()
                    self.oww.reset()
                    time.sleep(0.3)
        mic.stop_stream()
        mic.close()
        pa.terminate()

    def check_activated(self):
        if self._activated.is_set():
            self._activated.clear()
            return True
        return False


# ---------------------------------------------------------------------------
# EchoRobot
# ---------------------------------------------------------------------------
class EchoRobot:
    def __init__(self):
        print("Initializing Echo Robot Systems...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # FIX 2: Thread-safe context access.
        # visual_context is written by the main thread (YOLO) and read by the
        # audio worker thread (LLM). Without a lock, reads can observe a partial
        # list reassignment under CPython's GIL semantics.
        self._context_lock  = threading.Lock()
        self._visual_context = []  # always access via the property below

        # 1. Vision
        print("Loading Vision Module (YOLO)...")
        self.vision_model = YOLO(VISION_MODEL_PATH)

        # 2. STT
        print("Loading Audio Module (Whisper)...")
        self.stt_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            WHISPER_MODEL_PATH,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to(self.device)
        self.stt_processor = AutoProcessor.from_pretrained(WHISPER_MODEL_PATH)

        # 3. LLM — base model only, no LoRA adapter
        print("Loading Brain Module (Llama 3.2 3B — base, no LoRA)...")
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            LLM_BASE_MODEL,
            device_map="auto",
        )
        self.llm_model.eval()
        self.llm_tokenizer = AutoTokenizer.from_pretrained(LLM_BASE_MODEL)
        self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

        # 4. Gesture
        print("Loading Gesture Module (MediaPipe)...")
        self.gesture_detector = GestureDetector()

        # 5. Wake word
        print("Loading Wake Word Module (openwakeword)...")
        self.wakeword = WakeWordListener(sensitivity=0.5)

        # 6. TTS — loaded once to avoid ONNX reload penalty per utterance
        print("Loading TTS Voice (Piper)...")
        self.tts_voice = PiperVoice.load(JARVIS_VOICE_MODEL, config_path=JARVIS_VOICE_CONFIG)

        # 7. Threading
        self.audio_trigger  = queue.Queue()
        self.response_queue = queue.Queue()
        self.robot_state    = "IDLE"

        # 8. Rotating logger
        log_handler = RotatingFileHandler(
            'robot_history.log', maxBytes=5 * 1024 * 1024, backupCount=3)
        logging.basicConfig(handlers=[log_handler], level=logging.INFO,
                            format='%(asctime)s | %(message)s')
        self.logger = logging.getLogger('echo')

        # VRAM report
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024 ** 3
            total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            pct   = alloc / total * 100
            color = Fore.RED if pct > 90 else Fore.YELLOW if pct > 75 else Fore.GREEN
            print(f"{color}[VRAM] {alloc:.2f}GB / {total:.2f}GB ({pct:.0f}% used){Style.RESET_ALL}")
            if pct > 90:
                print(f"{Fore.RED}{Style.BRIGHT}[VRAM WARNING] >90% — high OOM risk!{Style.RESET_ALL}")

        print(f"{Fore.GREEN}{Style.BRIGHT}Robot Systems Online!{Style.RESET_ALL}")
        print(f"  Say {Fore.CYAN}'Hey Echo'{Style.RESET_ALL} or press {Fore.CYAN}'s'{Style.RESET_ALL} to speak")
        print(f"  Press {Fore.YELLOW}'g'{Style.RESET_ALL} for gesture, {Fore.RED}'q'{Style.RESET_ALL} to quit")

    # ------------------------------------------------------------------
    # Thread-safe visual_context property
    # ------------------------------------------------------------------
    @property
    def visual_context(self):
        with self._context_lock:
            return list(self._visual_context)  # return a shallow copy

    @visual_context.setter
    def visual_context(self, value):
        with self._context_lock:
            self._visual_context = value

    # ------------------------------------------------------------------
    # Vision
    # ------------------------------------------------------------------
    def get_visual_context(self, frame):
        h, w    = frame.shape[:2]
        results = self.vision_model(frame, conf=YOLO_CONF_THRESHOLD, verbose=False)
        items   = []

        for r in results:
            for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                name = self.vision_model.names[int(cls)]
                x1, y1, x2, y2 = box.tolist()
                cx = (x1 + x2) / 2

                h_pos = "left" if cx < w * 0.33 else ("center" if cx < w * 0.66 else "right")

                area_ratio = ((x2 - x1) * (y2 - y1)) / (w * h)
                dist = "close" if area_ratio > 0.3 else ("nearby" if area_ratio > 0.1 else "far")

                color_map = {"close": (0, 0, 255), "nearby": (0, 255, 255), "far": (0, 255, 0)}
                color = color_map[dist]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = f"{name} {conf:.0%}"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (int(x1), int(y1) - lh - 6),
                              (int(x1) + lw, int(y1)), color, -1)
                cv2.putText(frame, label, (int(x1), int(y1) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                items.append(f"{name} ({h_pos}, {dist}, conf:{conf:.2f})")

        return items

    # ------------------------------------------------------------------
    # Audio — VAD recording
    # ------------------------------------------------------------------
    def record_with_vad(self, sample_rate=16000, silence_threshold=1.5, max_duration=8):
        vad            = webrtcvad.Vad(2)
        chunk_duration = 0.03
        chunk_size     = int(sample_rate * chunk_duration)
        frames         = []
        silence_chunks = 0
        max_silence    = int(silence_threshold / chunk_duration)
        max_chunks     = int(max_duration / chunk_duration)
        speaking       = False

        print(f"\n{Fore.CYAN}[LISTENING]{Style.RESET_ALL} Speak now... "
              f"(auto-stops after {silence_threshold}s silence, max {max_duration}s)")

        try:
            with sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16') as stream:
                while len(frames) < max_chunks:
                    chunk, _    = stream.read(chunk_size)
                    chunk_bytes = chunk.tobytes()
                    is_speech   = vad.is_speech(chunk_bytes, sample_rate)

                    if is_speech:
                        speaking       = True
                        silence_chunks = 0
                        frames.append(chunk)
                    elif speaking:
                        silence_chunks += 1
                        frames.append(chunk)
                        if silence_chunks > max_silence:
                            print(f"{Fore.GREEN}[VAD]{Style.RESET_ALL} "
                                  f"Speech ended ({len(frames) * chunk_duration:.1f}s captured)")
                            break
        except Exception as e:
            print(f"{Fore.RED}[VAD ERROR]{Style.RESET_ALL} Recording failed: {e}")
            return np.array([], dtype=np.float32)

        if not frames:
            print(f"{Fore.YELLOW}[VAD]{Style.RESET_ALL} No speech detected.")
            return np.array([], dtype=np.float32)

        audio  = np.concatenate(frames).flatten().astype(np.float32)
        audio /= 32768.0
        return audio

    # ------------------------------------------------------------------
    # LLM — safety gate + inference
    # ------------------------------------------------------------------
    def _contains_movement_intent(self, transcript: str) -> bool:
        """
        Pre-LLM safety gate: returns True only if the transcript contains an
        explicit movement keyword. This prevents ambiguous inputs (e.g. 'move
        to court', 'go away from here') from being sent to the LLM with the
        expectation of a movement command.

        If False, the LLM prompt is still used, but parse_llm_output() will
        enforce NONE as the command — this gate is an additional audit layer.
        """
        words = set(re.findall(r'\b\w+\b', transcript.lower()))
        return bool(words & MOVEMENT_TRIGGER_WORDS)

    def query_brain(self, visual_context, user_input):
        try:
            print("[BRAIN] Analyzing visual context and query...")

            has_movement_intent = self._contains_movement_intent(user_input)
            movement_note = (
                "The user input contains explicit movement keywords. A movement command MAY be appropriate."
                if has_movement_intent else
                "The user input contains NO movement keywords. The <command> MUST be NONE."
            )

            # Format visual context as an explicit numbered list with count.
            # Giving the model the count directly prevents it from miscounting,
            # and the strict "only these objects" instruction reduces hallucination.
            if visual_context:
                obj_lines = "\n".join(f"  {i+1}. {obj}" for i, obj in enumerate(visual_context))
                context_block = (
                    f"Detected objects ({len(visual_context)} total):\n{obj_lines}\n\n"
                    f"STRICT RULE: Your response MUST only reference the {len(visual_context)} "
                    f"object(s) listed above. Never mention any object not in this list."
                )
            else:
                context_block = (
                    "Detected objects: none\n\n"
                    "STRICT RULE: You cannot see anything. Tell the user there are no objects visible."
                )

            # Use ChatML format — correct for Llama-3.2-3B-Instruct (unsloth variant).
            # The previous Alpaca ### Instruction / ### Response format caused the model
            # to reproduce prompt template text verbatim as its output.
            system_prompt = (
                "You are Echo, a robot assistant with a camera and motor control. "
                "You MUST reply using ONLY this exact XML format — no text outside the tags:\n\n"
                "<response>A natural, complete spoken sentence based only on detected objects.</response>\n"
                "<command>FORWARD|BACKWARD|LEFT|RIGHT|STOP|NONE</command>\n\n"
                "COMMAND RULES:\n"
                "- FORWARD  : 'move forward', 'go forward', 'advance'\n"
                "- BACKWARD : 'go back', 'reverse', 'retreat'\n"
                "- LEFT     : 'turn left', 'go left'\n"
                "- RIGHT    : 'turn right', 'go right'\n"
                "- STOP     : 'stop', 'halt'\n"
                "- NONE     : questions, greetings, descriptions, anything ambiguous\n\n"
                f"MOVEMENT GUIDANCE: {movement_note}"
            )

            user_message = (
                f"{context_block}\n\n"
                f"User said: {user_input}\n\n"
                "Reply in the required XML format."
            )

            # Apply the model's built-in chat template so special tokens are
            # handled correctly (e.g. <|im_start|>, <|im_end|>, <|eot_id|>).
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ]
            prompt = self.llm_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = self.llm_tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.llm_model.generate(
                    **inputs,
                    max_new_tokens=120,
                    max_length=None,       # suppress the max_length config warning
                    do_sample=True,
                    temperature=0.1,       # lower = more deterministic, less hallucination
                    top_p=0.9,
                    pad_token_id=self.llm_tokenizer.eos_token_id,
                    eos_token_id=self.llm_tokenizer.eos_token_id,
                )

            # Decode only the newly generated tokens (strip the prompt).
            input_len     = inputs["input_ids"].shape[1]
            new_tokens    = outputs[0][input_len:]
            return self.llm_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"\n{Fore.RED}{Style.BRIGHT}[OOM]{Style.RESET_ALL} VRAM full — cleared cache")
            self.speak_async("Memory is full, please try again.")
            return None
        except Exception as e:
            print(f"\n{Fore.RED}[BRAIN ERROR]{Style.RESET_ALL} LLM inference failed: {e}")
            self.speak_async("Sorry, I could not process that.")
            return None

    # ------------------------------------------------------------------
    # FIX 3: Stricter parse_llm_output — NONE is the safe default
    # ------------------------------------------------------------------
    def parse_llm_output(self, raw_output: str, transcript: str = ""):
        """
        Parse XML tags from LLM output.

        Safety contract (two layers):
        1. Structural: if <command> tag is missing, malformed, or unrecognised → NONE.
        2. Semantic:   if the original transcript contains no movement keyword,
                       hard-override any movement command to NONE — regardless of
                       what the LLM decided. This prevents spatial context leakage
                       where the LLM maps positional words (left/right in visual
                       context) onto the command field for non-movement queries.

        The second layer is the critical addition: it makes the safety guarantee
        independent of LLM compliance, which cannot be relied upon at 3B scale.
        """
        def extract_tag(text, tag):
            match = re.search(f'<{tag}>(.*?)</{tag}>', text, re.DOTALL | re.IGNORECASE)
            return match.group(1).strip() if match else None

        response_text = extract_tag(raw_output, "response") or raw_output
        raw_command   = extract_tag(raw_output, "command") or ""

        # Layer 1 — validate command is a known value
        command = raw_command.strip().upper()
        if command not in VALID_COMMANDS:
            if command:
                print(f"{Fore.YELLOW}[PARSE WARN]{Style.RESET_ALL} "
                      f"Unexpected command '{command}' → defaulting to NONE")
            command = "NONE"

        # Layer 2 — hard override: non-movement input MUST produce NONE command.
        # The LLM prompt hint alone is insufficient because small models leak
        # spatial vocabulary from visual context (e.g. "left" in context → LEFT).
        if command != "NONE" and not self._contains_movement_intent(transcript):
            print(f"{Fore.YELLOW}[SAFETY GATE]{Style.RESET_ALL} "
                  f"Command '{command}' blocked — no movement keyword in transcript.")
            command = "NONE"

        intent = "QUERY"
        return intent, response_text, command

    # ------------------------------------------------------------------
    # TTS
    # ------------------------------------------------------------------
    def _clean_speech_text(self, text):
        match = re.search(r'<response>(.*?)</response>', text, re.DOTALL)
        if match:
            return match.group(1).strip()

        text = re.sub(r'<command>.*?</command>', '', text, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', '', text)

        lines = []
        for line in text.split('\n'):
            s = line.strip()
            if not s or s.startswith('###'):
                continue
            if re.match(r'^(Motor Command|Command|Intent)\s*:', s, re.I):
                continue
            if re.match(r'^(FORWARD|BACKWARD|LEFT|RIGHT|STOP|NONE|QUERY|COMMAND|GESTURE_COMMAND)$',
                        s, re.I):
                continue
            lines.append(s)

        result = re.sub(r'\s+', ' ', ' '.join(lines)).strip()
        return "" if re.match(r'^\d+$', result) else result

    def speak_async(self, text):
        clean_text = self._clean_speech_text(text)
        if not clean_text:
            return

        def _speak():
            try:
                print(f"{Fore.BLUE}{Style.BRIGHT}[🔊 SPEAKING]{Style.RESET_ALL} {clean_text}")
                audio_bytes = b''
                for chunk in self.tts_voice.synthesize(clean_text):
                    audio_bytes += chunk.audio_int16_bytes
                if audio_bytes:
                    audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
                    sd.play(audio_data, samplerate=self.tts_voice.config.sample_rate)
                    sd.wait()
            except Exception as e:
                print(f"[TTS ERROR] Speech failed: {e}")

        threading.Thread(target=_speak, daemon=True).start()

    # ------------------------------------------------------------------
    # Gesture — deterministic, bypasses LLM entirely
    # ------------------------------------------------------------------
    def handle_gesture_command(self, gesture):
        command  = GESTURE_COMMAND_MAP.get(gesture)
        response = GESTURE_RESPONSE_MAP.get(gesture, "Gesture not recognized.")
        if command:
            self.send_motor_command(command)
            self.speak_async(response)
            print(f"\n{Fore.YELLOW}{'='*50}")
            print(f"{Fore.YELLOW}{Style.BRIGHT}[GESTURE]{Style.RESET_ALL} "
                  f"{gesture} → {Fore.RED}{command}")
            print(f"{Fore.YELLOW}{'='*50}\n")
            self.logger.info(
                f"Intent: GESTURE_COMMAND | Source: gesture | "
                f"Context: {self.visual_context} | Input: gesture:{gesture} | "
                f"Response: {response} | Command: {command}")
        else:
            print(f"{Fore.YELLOW}[GESTURE]{Style.RESET_ALL} '{gesture}' — no mapping found.")

    def send_motor_command(self, command):
        print(f"{Fore.MAGENTA}[MOTOR]{Style.RESET_ALL} >> {command} (simulated — ESP32 not connected)")

    # ------------------------------------------------------------------
    # Audio pipeline worker thread
    # ------------------------------------------------------------------
    def _audio_pipeline_worker(self):
        while True:
            trigger_source = self.audio_trigger.get()
            if trigger_source is None:
                break

            try:
                self.robot_state = "LISTENING"
                audio_data = self.record_with_vad()

                if audio_data.size == 0:
                    self.robot_state = "IDLE"
                    continue

                self.robot_state = "THINKING"
                print("[AUDIO CAPTURED] Transcribing...")

                input_features = self.stt_processor(
                    audio_data, sampling_rate=16000, return_tensors="pt"
                ).input_features.to(self.device).to(torch.float16)

                predicted_ids = self.stt_model.generate(
                    input_features,
                    attention_mask=torch.ones(input_features.shape[:2],
                                             dtype=torch.long, device=self.device),
                    language="en",
                    task="transcribe",
                    suppress_tokens=[],
                    forced_decoder_ids=None,
                )
                transcript = self.stt_processor.batch_decode(
                    predicted_ids, skip_special_tokens=True
                )[0].strip()

                if not transcript:
                    self.robot_state = "IDLE"
                    continue

                print(f"\n{Fore.CYAN}{Style.BRIGHT}USER ({trigger_source}):{Style.RESET_ALL} {transcript}")

                current_gesture = self.gesture_detector.get_gesture()
                user_input      = f"voice: {transcript}"
                if current_gesture != "none":
                    user_input += f" | gesture: {current_gesture}"

                # FIX 2: Read visual_context via the thread-safe property (returns a copy).
                context_snapshot = self.visual_context

                raw_output = self.query_brain(context_snapshot, user_input)
                if raw_output is None:
                    self.robot_state = "IDLE"
                    continue

                intent, resp_text, cmd_text = self.parse_llm_output(raw_output, transcript)
                self.response_queue.put((intent, resp_text, cmd_text, user_input, trigger_source))

            except Exception as e:
                print(f"{Fore.RED}[PIPELINE ERROR]{Style.RESET_ALL} {e}")
                self.robot_state = "IDLE"

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self):
        cap = cv2.VideoCapture(0)
        self.wakeword.start()
        audio_worker = threading.Thread(target=self._audio_pipeline_worker, daemon=True)
        audio_worker.start()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # FIX 2: Write via the thread-safe setter.
            self.visual_context = self.get_visual_context(frame)

            gesture = self.gesture_detector.process_frame(frame)

            if self.wakeword.check_activated() and self.robot_state == "IDLE":
                self.audio_trigger.put("wake_word")

            try:
                intent, resp_text, cmd_text, user_input, source = self.response_queue.get_nowait()

                intent_colors = {"QUERY": Fore.CYAN, "COMMAND": Fore.MAGENTA,
                                 "GESTURE_COMMAND": Fore.YELLOW}
                intent_color  = intent_colors.get(intent, Fore.WHITE)

                print(f"\n{Fore.YELLOW}{'='*50}")
                print(f"{intent_color}{Style.BRIGHT}[{intent}]{Style.RESET_ALL} via {source}")
                print(f"{Fore.GREEN}{Style.BRIGHT}RESPONSE:{Style.RESET_ALL} {resp_text}")
                print(f"{Fore.MAGENTA}{Style.BRIGHT}COMMAND:{Style.RESET_ALL} {Fore.RED}{cmd_text}")
                print(f"{Fore.YELLOW}{'='*50}\n")

                self.robot_state = "SPEAKING"
                if resp_text and resp_text != "N/A":
                    self.speak_async(resp_text)

                # Only execute movement commands — log everything
                if cmd_text != "NONE":
                    self.send_motor_command(cmd_text)

                self.logger.info(
                    f"Intent: {intent} | Source: {source} | "
                    f"Context: {self.visual_context} | "
                    f"Input: {user_input} | Response: {resp_text} | Command: {cmd_text}")

                threading.Timer(1.0, self._set_idle).start()

            except queue.Empty:
                pass

            # HUD
            state_colors = {
                "IDLE": (128, 128, 128), "LISTENING": (0, 255, 255),
                "THINKING": (255, 165, 0), "SPEAKING": (0, 200, 255),
            }
            state_color = state_colors.get(self.robot_state, (255, 255, 255))
            ctx_snapshot = self.visual_context  # one lock acquisition for display
            cv2.putText(frame, f"Context: {', '.join(ctx_snapshot)}", (10, 30),
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
                self.audio_trigger.put("manual")
            elif key == ord('g'):
                current_gesture = self.gesture_detector.get_gesture()
                if current_gesture != "none" and self.robot_state == "IDLE":
                    self.handle_gesture_command(current_gesture)
                else:
                    print(f"{Fore.YELLOW}[GESTURE]{Style.RESET_ALL} No gesture detected.")

        self.wakeword.stop()
        self.audio_trigger.put(None)
        self.gesture_detector.close()
        cap.release()
        cv2.destroyAllWindows()

    def _set_idle(self):
        self.robot_state = "IDLE"


if __name__ == "__main__":
    robot = EchoRobot()
    robot.run()