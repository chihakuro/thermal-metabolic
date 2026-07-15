# 🌡️ Thermal-Metabolic Edge AI (Cognitive HVAC System)

A distributed Edge AI solution that transforms conventional HVAC systems into personalized climate zones. By processing 100% offline on edge devices using low-resolution thermal sensors, it ensures physiological comfort without compromising privacy.

## 🎯 Problem & Vision

Traditional HVAC systems in open offices or halls only measure ambient air temperature at a single point on the wall. They cannot detect if occupants are sweating or shivering. Using RGB cameras to monitor humans is a severe violation of workplace privacy.

**The Solution:** Utilizing ultra-low-resolution thermal sensor arrays (MLX90640 - 24x32 pixels) combined with a Micro-CNN and Thermal-Metabolic logic. The system automatically analyzes the "thermal footprint" of individuals to dynamically direct cool airflow or shut off vents in real-time, completely anonymously.

## ✨ Core Features

* **🔒 Privacy-First:** No optical RGB cameras are used. The algorithm only processes matrices of physical temperature values (Celsius).
* **🪶 Micro-CNN Architecture:** A custom-built, ultra-lightweight neural network with only ~26,000 parameters (model size ~100 KB).
* **⚡ Industrial Quantization (ONNX):** The model is optimized and executed entirely via ONNX Runtime, stripping away heavy PyTorch dependencies for the production environment.
* **🎛️ Multi-Zone HVAC Dashboard:** Analyzes and executes parallel decisions for multiple independent climate zones within a large commercial space.
* **📦 Containerized Edge Deployment:** Fully packaged with Docker, featuring an ultra-slim footprint (Pure NumPy + ONNX).

## 🛠️ Tech Stack

* **AI Framework:** PyTorch (for Model Design & Training)
* **Edge Inference:** ONNX Runtime, NumPy, Pandas (No-Torch Pipeline)
* **Containerization:** Docker

## 🏗️ System Architecture

The inference pipeline is divided into two sequential stages to ensure maximum accuracy and practical applicability:

1. **👁️ Spatial AI Detector:** The Micro-CNN scans the 24x32 thermal matrix to detect human presence and locate their coordinates, filtering out static heat sources like laptops or lamps.
2. **🧬 Metabolic Analyzer:** Applies physiological rules to the detected human regions to extract their Thermal Footprint (Core Body Temp & Average Surface Temp).

### Metabolic Decision Tree

| Core Temp | Surface (Avg) Temp | Physiological State | HVAC Command |
| --- | --- | --- | --- |
| > 34.5°C | > 32.0°C | HOT (Active/Sweating) | MAX COOLING |
| < 32.0°C | < 29.5°C | COLD (Thick clothing/Heat loss) | CLOSE VENTS / HEAT |
| 32.0°C - 34.5°C | 29.5°C - 32.0°C | COMFORTABLE (Stable) | MAINTAIN CURRENT |
| N/A | N/A | EMPTY (No humans detected) | SYSTEM OFF |

## 📁 Repository Structure

```text
thermal-metabolic/
├── models/
│   ├── micro_cnn.py         # Original PyTorch model architecture
│   ├── thermal_model.pth    # Model weights (Float32)
│   └── thermal_model.onnx   # Production-ready ONNX model
├── data_loader.py           # Training data pipeline
├── export_model.py          # Script for ONNX quantization and export
├── multi_zone_stream.py     # Core inference script (Pure ONNX Dashboard)
├── requirements.txt         # Production dependencies
├── Dockerfile               # Edge AI container configuration
└── README.md

```

*(Note: The raw `.csv` training data is excluded from this repository to optimize storage).*

## 🚀 Quick Start

The system is designed to run instantly on any Docker-supported environment without complex configurations.

**Step 1: Build the Docker Image**
From the root directory of the project, run:

```bash
docker build -t thermal-edge-app:production .

```

**Step 2: Launch the Multi-Zone Dashboard**
Run the container to monitor the real-time simulation:

```bash
docker run --rm thermal-edge-app:production

```

The console will immediately display a real-time Multi-Zone dashboard (Entrance, Center, Corner) with HVAC decisions updated every 0.5 seconds.

## 🗺️ Roadmap (Phase 2)

* **Hardware Deployment:** Translate the inference logic to C++ for direct flashing onto ESP32 / RPi microcontrollers.
* **I2C Integration:** Replace CSV simulations with live I2C data streaming directly from the MLX90640 sensor.
* **IoT Connectivity:** Integrate MQTT protocol to broadcast JSON telemetry data to a central Home Assistant server.