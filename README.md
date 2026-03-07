# Echo: Context-Aware Interactive Robot Assistant

This repository contains the software stack for the **Context-Aware Interactive Robot Assistant (Echo)**, employing a hybrid edge-server architecture involving a Raspberry Pi 4, an ESP32 microcontroller, and a Laptop server.

## System Architecture

- **Laptop (Server)**: Computes heavy AI tasks such as YOLO11n for object detection, Faster-Whisper for Speech-To-Text (STT), and Ollama (Llama 3.2 3B) for Large Language Model processing.
- **Raspberry Pi 4 (Edge)**: Handles audio I/O using Piper TTS and passes commands to the ESP32.
- **ESP32 (Edge/Motors)**: Manages physical interactions via a motor control unit (L298N) and obstacle avoidance (HC-SR04).

## Setup
The primary dependencies are managed using Python 3.10 and `pip`. Setup and run logs can be found in `robot-assistant/logs`.
