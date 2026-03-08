# Echo: Context-Aware Interactive Robot Assistant

This repository contains the software stack for Echo, a context-aware interactive robot assistant. The system utilizes a hybrid edge-server architecture to balance high-performance AI processing with real-time hardware control.

## System Architecture

The project is distributed across three primary hardware components:

### Laptop (Server)
The laptop serves as the central processing unit for computationally intensive AI tasks, utilizing an NVIDIA GPU for accelerated inference. Key components include:
*   **Object Detection:** Fine-tuned YOLO11n (`yolo11n-gestures.pt`) for real-time hand gesture recognition and object identification.
*   **Speech-to-Text (STT):** Fine-tuned Whisper-small for high-accuracy English-only command recognition, eliminating hallucinations and Indonesian fallback.
*   **Large Language Model (LLM):** Llama 3.2 3B with a custom LoRA adapter for structured command extraction and intelligent response generation.
*   **Integrated Control:** `scripts/robot_control_v2.py` as the primary orchestration script.

### Raspberry Pi 4 (Edge)
The Raspberry Pi handles user interaction and acts as a bridge to the physical hardware:
*   **Audio Output:** Piper TTS for high-quality, low-latency text-to-speech synthesis.
*   **Serial Bridge:** Manages communication with the ESP32 via UART/Serial.
*   **Interaction Logic:** Coordinates local sensors and audio I/O.

### ESP32 (Hardware Control)
The ESP32 manages the physical components of the robot:
*   **Motor Control:** L298N motor driver interface for movement.
*   **Obstacle Avoidance:** Integration with HC-SR04 ultrasonic sensors.

## Project Structure

The codebase is organized into the following directory structure:

*   `scripts/training/`: Fine-tuning scripts for Whisper, YOLO, and Llama.
*   `scripts/evaluation/`: Tools for verifying model performance.
*   `robot-assistant/models/`: Storage for fine-tuned weights (Whisper, YOLO, LLM adapter).
*   `robot-assistant/data/`: Datasets used for fine-tuning.

## Installation and Setup

### Prerequisites
*   Windows OS (Server) / Linux (Raspberry Pi).
*   Python 3.10 or higher.
*   NVIDIA GPU with CUDA support (RTX 30 series recommended).
*   Hugging Face Token configured in `.env` (for LLM fine-tuning).

### Setup Steps
1.  Clone the repository.
2.  Create and activate a virtual environment: `python -m venv robot-env`.
3.  Install dependencies: `pip install -r requirements.txt`.
4.  Configure `.env` with your `ROBOFLOW_API_KEY` and `HF_TOKEN`.

## Running the Robot

To launch the integrated multimodal robot control system:
```powershell
.\robot-env\Scripts\python.exe scripts/robot_control_v2.py
```
- **Vision**: Continuous monitoring of context and hand gestures.
- **Voice**: Press `s` to trigger the Whisper-small listener.
- **Brain**: Llama 3.2 3B processes the combined input to decide the next action.

## Fine-Tuning Results
- **STT (Whisper)**: 90% accuracy on English-only commands.
- **Vision (YOLO)**: 91.4% mAP for hand gestures.
- **LLM (Llama)**: 100% adherence to structured XML command output.

