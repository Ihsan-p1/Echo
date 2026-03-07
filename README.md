# Echo: Context-Aware Interactive Robot Assistant

This repository contains the software stack for Echo, a context-aware interactive robot assistant. The system utilizes a hybrid edge-server architecture to balance high-performance AI processing with real-time hardware control.

## System Architecture

The project is distributed across three primary hardware components:

### Laptop (Server)
The laptop serves as the central processing unit for computationally intensive AI tasks, utilizing an NVIDIA GPU for accelerated inference. Key components include:
*   **Object Detection:** YOLO11n for real-time identification of indoor and computer lab objects.
*   **Speech-to-Text (STT):** Faster-Whisper (medium model) for high-accuracy transcription of English and Indonesian commands.
*   **Large Language Model (LLM):** Ollama running Llama 3.2 3B for natural language understanding and command extraction.
*   **API Server:** A Flask-based backend to coordinate communication between the edge devices and AI models.

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

*   `robot-assistant/server/`: Backend logic for vision, language, and STT.
*   `robot-assistant/rpi/`: Real-time interaction logic for the Raspberry Pi.
*   `robot-assistant/logs/`: Detailed logs for setup and evaluation phases.
*   `Docs/`: System Design Documents and project requirements (ignored by version control).
*   `data/` and `models/`: Directories for storing fine-tuning datasets and optimized model weights (ignored by version control).

## Installation and Setup

### Prerequisites
*   Windows OS (Server) / Linux (Raspberry Pi).
*   Python 3.10 or higher.
*   NVIDIA GPU with CUDA support (for Server).
*   Ollama installed and configured with Llama 3.2 3B.

### Setup Steps
1.  Clone the repository.
2.  Create and activate a virtual environment: `python -m venv robot-env`.
3.  Install dependencies: `pip install -r requirements.txt` (or install core packages manually as detailed in setup logs).
4.  Configure environment variables, specifically `ROBOFLOW_API_KEY` if dataset downloads are required.

## Evaluation and Fine-Tuning
The project follows a multi-phase approach:
1.  **Phase 1 (Setup):** Installation and verification of all core models.
2.  **Phase 2 (Evaluation):** Establishing performance baselines for STT, Vision, and LLM components.
3.  **Phase 3 (Fine-Tuning):** Targeted optimization of models that fail to meet baseline performance criteria.

Detailed evaluation metrics and pass/fail triggers are documented in the project logs.

