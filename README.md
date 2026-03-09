# Echo: Context-Aware Interactive Robot Assistant

This repository contains the software stack for **Echo**, a context-aware interactive robot assistant. The system utilizes a hybrid edge-server architecture to balance high-performance AI processing with real-time hardware control, now featuring a fully threaded non-blocking pipeline.

## 🚀 Key Features (v2.0)

*   **Wake Word Activation**: Background listening for "Hey Jarvis" via `openwakeword`.
*   **Threaded Architecture**: Separate threads for Vision, Wake Word, and Audio Pipeline to ensure 0ms lag in gesture/vision processing.
*   **Spatial Vision**: Distance-aware object detection (Red/Yellow/Green bboxes based on proximity) with 3D-like context (left/center/right).
*   **VAD Recording**: Voice Activity Detection (`webrtcvad`) that automatically stops recording when you finish speaking.
*   **Multimodal Intent**: A unified LLM brain that parses Voice and Gesture simultaneously into structured XML `<intent>`, `<response>`, and `<command>`.
*   **State HUD**: Real-time on-screen status (IDLE, LISTENING, THINKING, SPEAKING).

## ## System Architecture

The project is distributed across three primary hardware components:

### Laptop (Server)
The laptop serves as the central processing unit for AI tasks, utilizing NVIDIA GPU (CUDA) for acceleration:
*   **Vision Engine**: YOLO11n for environment context and object triangulation.
*   **Gesture Engine**: MediaPipe HandLandmarker for precise background gesture recognition (Stop, Forward, etc.).
*   **STT (Speech-to-Text)**: Fine-tuned Whisper-small with VAD for silent-cutoff recording.
*   **Brain (LLM)**: Llama 3.2 3B + LoRA adapter for structured intent parsing.
*   **TTS (Text-to-Speech)**: Optimized Piper TTS (loaded once into VRAM) for Jarvis-like responses.

### Raspberry Pi 4 (Edge)
Acts as the hardware bridge:
*   **Serial Bridge**: UART communication with ESP32.
*   **Edge Audio**: Local speaker output for generated speech.

### ESP32 (Hardware Control)
*   **Motor Control**: Direct L298N interface.
*   **Sensors**: Ultrasonic HC-SR04 for low-level obstacle override.

## ## Project Structure
*   `scripts/robot_control_v2.py`: The main threaded orchestration script.
*   `robot-assistant/models/`: Fine-tuned weights (YOLO, Whisper, LLM adapter).
*   `robot_history.log`: Rotating logs (5MB, 3 backups) documenting robot decisions.

## ## Installation & Usage

### 1. Requirements
*   Python 3.10+
*   NVIDIA GPU (CUDA-ready)
*   High-quality Microphone & Camera

### 2. Launch
```powershell
# Activate environment and run
.\robot-env\Scripts\python.exe scripts/robot_control_v2.py
```

### 3. Controls
*   **Wake Word**: Just say **"Hey Jarvis"** to start speaking.
*   **Manual Trigger**: Press `s` to force open the microphone.
*   **Gesture Mode**: Press `g` to snapshot current hand gesture into the brain.
*   **Quit**: Press `q` to exit.

## ## Performance Stats
*   **VRAM Usage**: ~2.6GB (Green zone on 6GB+ GPUs).
*   **STT Accuracy**: 90%+ on English commands.
*   **Response Latency**: ~500ms startup saving due to Piper pre-loading.


