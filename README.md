# ✈️ B787-ignition-digital-twin

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.103.0-009688.svg)](https://fastapi.tiangolo.com/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0.0-ff69b4.svg)](https://lightgbm.readthedocs.io/)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-20BEFF.svg)](https://www.kaggle.com/datasets/mohammedbellosani/boeing-787-ignition-digital-twin-and-phm-dataset)
[![Status](https://img.shields.io/badge/Status-Under_Review_(RESS)-success.svg)]()

> **Codebase for the manuscript:** *"Digital Twin Framework for Gas Turbine Ignition Prognostics and Real-Time Fault Injection"* (Submitted to *Reliability Engineering & System Safety*).

---

## 📖 Overview
The transition from reactive to predictive maintenance in commercial aviation is severely hindered by the scarcity of run-to-failure data, particularly for highly reliable components like Full Authority Digital Engine Control (FADEC) ignition systems. 

This repository contains the full source code for a **closed-loop Prognostics and Health Management (PHM) Digital Twin** developed for the Boeing 787-8 (ATA 74) ignition system. By bridging live macro-physics telemetry from a flight simulator with synthesized micro-electrical degradation data, this framework deterministically predicts Remaining Useful Life (RUL) and issues real-time Condition-Based Maintenance (CBM) directives.

![System Architecture](https://raw.githubusercontent.com/Sm-bello/b787-ignition-digital-twin/main/architecture.png)
*(Note: Upload your `architecture.png` to the root of this repo to render this image!)*

---

## 🛠️ System Methodology & Architecture

Our framework overcomes the aviation industry's data scarcity problem through a four-pillar approach:

### 1. Physics-Informed Synthetic Data Generation
Because catastrophic failure data for commercial aircraft is practically non-existent, we engineered a thermodynamic ODE solver to model the deep electrical discharge physics of a High-Energy Capacitor-Discharge Igniter. We generated a balanced **27,000-sample dataset** simulating 11 extreme, out-of-distribution fault modes (e.g., exciter box capacitance drops, electrode erosion, high-altitude flameouts).
* 📊 **[Access the full 100% Usability Dataset on Kaggle here.](https://www.kaggle.com/datasets/mohammedbellosani/boeing-787-ignition-digital-twin-and-phm-dataset)**

### 2. LightGBM Prognostic Engine
To resolve stochastic conflicts between separate classification and regression models, we adopted a **Single-Model Architecture**. A LightGBM Regressor was trained on 19 scaled features to map deep electrical wear directly to Remaining Useful Life (RUL).
* **Performance:** R² of 0.87 | Mean Absolute Error (MAE) of 39 flight cycles.

### 3. Asynchronous FastAPI Microservice
To ensure the high-frequency (10Hz) UDP telemetry bridge from the flight simulator is not bottlenecked by AI inference times, the LightGBM model is hosted on a decoupled local FastAPI microservice. This ensures sub-second latency for real-time inference.

### 4. "Deep Feature Spoofing" Validation (God Mode)
Validating PHM systems against mid-flight catastrophic failures is prohibitively dangerous. We built a custom Tkinter GUI that acts as a "Man-in-the-Middle" telemetry spoofer. When a user triggers an anomaly, the system actively poisons the intermediate mathematical arrays, allowing us to safely evaluate the AI's prognostic accuracy without destabilizing the underlying physical simulation environment.

![Deep Feature Spoofing in Action](https://raw.githubusercontent.com/Sm-bello/b787-ignition-digital-twin/main/spoof_in_action.png)
*(Note: Upload your `spoof_in_action.png` to the root of this repo to render this image!)*

---

## 📂 Repository Structure

```text
📦 b787-ignition-digital-twin
 ┣ 📂 api                  # FastAPI microservice for LightGBM inference
 ┣ 📂 dashboard            # Tkinter GUI (God Mode Spoofing Interface)
 ┣ 📂 models               # Serialized LightGBM models & scalers
 ┣ 📂 simulator_bridge     # UDP socket listeners for FlightGear XML protocols
 ┣ 📜 main.py              # Application entry point
 ┣ 📜 requirements.txt     # Python dependencies
 ┗ 📜 README.md
