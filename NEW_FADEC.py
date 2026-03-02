"""
FlightGear Full-Stack Digital Twin (Tkinter GUI + REST API + God Mode)
======================================================================
✨ FULL STACK EDITION:
- Hosts the FastAPI microservice on Port 8000 in a background thread.
- Runs the UDP telemetry listener in a background thread.
- Runs the Tkinter GUI on the main thread.
- Includes ALL 11 authentic Boeing 787-8 ATA 74 Fault Injections.
- Deep Feature Spoofing enabled for accurate RUL prediction matching.
- NO OMISSIONS. Fully expanded GUI code.
"""

import socket
import threading
import numpy as np
import tkinter as tk
from tkinter import messagebox
from collections import deque
import warnings
import sys
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from inference import RULPredictor

# Suppress annoying, harmless warnings from sklearn and lightgbm
warnings.filterwarnings("ignore", message="Skipping features without any observed values")
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------- Configuration ----------------------------
UDP_IP = "127.0.0.1"
UDP_PORT = 5550
BUFFER_SIZE = 1024
MODEL_DIR = "./rul_models_output"


# ============================================================================
# FASTAPI MICROSERVICE SETUP
# ============================================================================
app = FastAPI(title="B787 Ignition Digital Twin API", version="2.0")
api_predictor = None  # Instantiated inside the thread to avoid memory lock

class SensorData(BaseModel):
    spark_energy_J: float = 0.0
    spark_peak_voltage_V: float = 0.0
    spark_peak_current_A: float = 0.0
    spark_voltage_std_V: float = 0.0
    ignition_delay_s: float = 0.0
    ignition_success: int = 0
    max_EGT_rise_rate_Kps: float = 0.0
    time_to_stable_s: float = 0.0
    peak_chamber_pressure_bar: float = 0.0
    mean_chamber_pressure_bar: float = 0.0
    vibration_peak_amplitude_Pa: float = 0.0
    vibration_rms_Pa: float = 0.0
    vibration_dominant_freq_Hz: float = 0.0
    EGT_std_K: float = 0.0
    pressure_std_bar: float = 0.0
    combustion_efficiency: float = 0.0
    spark_efficiency: float = 0.0
    igniter_resistance_ohm: float = 0.0
    spark_decay_rate: float = 0.0

@app.post("/predict_health")
def predict_rul(data: SensorData):
    """Endpoint that receives telemetry and returns health status"""
    try:
        global api_predictor
        if not api_predictor:
            api_predictor = RULPredictor(model_dir=MODEL_DIR)
            
        features_dict = data.dict()
        result = api_predictor.predict(features_dict)
        
        if result['status'] == 'error':
            raise HTTPException(status_code=500, detail=result['error_message'])
            
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health_check")
def ping():
    """Simple endpoint to verify the API is online"""
    return {"status": "online", "message": "Ignition Digital Twin API is active."}

def run_fastapi():
    """Runs the Uvicorn server in a separate thread without blocking Tkinter"""
    config = uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="error")
    server = uvicorn.Server(config)
    # Disable signal handling so it doesn't crash the main Tkinter thread
    server.install_signal_handlers = lambda: None 
    server.run()


# ---------------------------- Fault Injection Definitions ----------------------------
FAULT_SCENARIOS = {
    'ground': {
        'hot_start': {
            'name': 'Hot Start (Fuel Accumulation)',
            'description': 'Excessive EGT spike during ground start',
            'risk': 'LOW',
            'modifications': {'spark_energy_override': 3.5, 'vibration_multiplier': 1.0}
        },
        'hung_start': {
            'name': 'Hung Start (Low RPM Stall)',
            'description': 'Engine fails to accelerate during start',
            'risk': 'LOW',
            'modifications': {'spark_energy_override': 3.5, 'vibration_multiplier': 1.0}
        },
        'misfire': {
            'name': 'Misfire / Igniter Failure',
            'description': 'Complete ignition system failure',
            'risk': 'MEDIUM',
            'modifications': {'spark_energy_override': 0.0, 'vibration_multiplier': 1.0}
        },
        'weak_spark': {
            'name': 'Degraded Igniter (Weak Spark)',
            'description': 'Reduced spark energy, slow ignition',
            'risk': 'LOW',
            'modifications': {'spark_energy_override': 1.2, 'vibration_multiplier': 1.2}
        },
        'exciter_box_failure': {
            'name': 'Exciter Box Failure (Capacitor Drop)',
            'description': 'Capacitor fails to hold full charge, low Joules',
            'risk': 'MEDIUM',
            'modifications': {'spark_energy_override': 0.8, 'vibration_multiplier': 1.0}
        },
        'igniter_fouling': {
            'name': 'Igniter Fouling (Carbon)',
            'description': 'Carbon deposits slow down ignition tracking',
            'risk': 'LOW',
            'modifications': {'spark_energy_override': 2.5, 'vibration_multiplier': 1.5}
        }
    },
    'airborne': {
        'engine_flameout': {
            'name': 'In-Flight Flameout',
            'description': 'Combustion extinction at altitude',
            'risk': 'HIGH',
            'modifications': {'spark_energy_override': 0.0, 'vibration_multiplier': 1.0}
        },
        'igniter_degradation': {
            'name': 'Continuous Igniter Wear',
            'description': 'Continuous ignition system wear during cruise',
            'risk': 'MEDIUM',
            'modifications': {'spark_energy_override': 2.1, 'vibration_multiplier': 2.5}
        },
        'electrode_erosion': {
            'name': 'Electrode Tip Erosion',
            'description': 'Physical wear on spark plug tip',
            'risk': 'MEDIUM',
            'modifications': {'spark_energy_override': 1.5, 'vibration_multiplier': 1.8}
        },
        'insulation_breakdown': {
            'name': 'Insulation Breakdown',
            'description': 'High altitude arcing and voltage drops',
            'risk': 'HIGH',
            'modifications': {'spark_energy_override': 1.0, 'vibration_multiplier': 2.0}
        },
        'high_altitude_relight': {
            'name': 'High-Altitude Relight Failure',
            'description': 'Igniter struggles at low air density',
            'risk': 'HIGH',
            'modifications': {'spark_energy_override': 1.8, 'vibration_multiplier': 1.5}
        }
    }
}


# ---------------------------- Main GUI Class ----------------------------
class DigitalTwinGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AEROTWIN | B787 Ignition Health Monitor [FULL STACK GOD MODE]")
        self.root.geometry("1450x850")
        self.root.configure(bg="#0b132b")

        # Load predictor
        print("[OK] Loading RULPredictor...")
        self.predictor = RULPredictor(model_dir=MODEL_DIR)

        # Shared data
        self.latest_fg = None
        self.latest_result = None
        self.lock = threading.Lock()

        # Feature engineering states
        self.egt_history = deque(maxlen=10)
        self.prev_time = None
        self.prev_egt_k = None

        # 🔥 FAULT INJECTION STATE
        self.active_fault = None
        self.fault_overrides = {}
        self.last_flight_phase = None
        self.test_offer_shown = False

        # Build UI
        self.build_ui()

        # Start API Thread
        self.api_thread = threading.Thread(target=run_fastapi, daemon=True)
        self.api_thread.start()
        print("[OK] FastAPI Microservice started on http://127.0.0.1:8000")

        # Start UDP listener thread
        self.running = True
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((UDP_IP, UDP_PORT))
        self.udp_thread = threading.Thread(target=self.udp_listener, daemon=True)
        self.udp_thread.start()

        # Start GUI update loop
        self.update_gui()

    # ---------------------------- UI Construction ----------------------------
    def build_ui(self):
        # Header Frame
        header_frame = tk.Frame(self.root, bg="#0b132b")
        header_frame.pack(fill="x", pady=10)

        # Header Text
        header_text = tk.Label(header_frame,
                          text="◆ AEROTWIN MRO MONITOR [GOD MODE]   |   BOEING 787-8   |   ATA 74 (IGNITION)",
                          bg="#0b132b", fg="#e63946",
                          font=("Courier", 14, "bold"), anchor="w", padx=20, pady=10)
        header_text.pack(side="left")

        # API Status Label
        api_status = tk.Label(header_frame, 
                              text="API MICROSERVICE: ONLINE (Port 8000) 🟢", 
                              bg="#0b132b", fg="#00ff41", 
                              font=("Courier", 10, "bold"))
        api_status.pack(side="right", padx=20)

        # Main content frame
        main = tk.Frame(self.root, bg="#0b132b")
        main.pack(fill="both", expand=True, padx=20, pady=10)

        # ---------- Left Panel: Telemetry ----------
        left = tk.Frame(main, bg="#0b132b", width=400)
        left.pack(side="left", fill="y", padx=(0, 20))

        # Engine 1 block
        eng1_label = tk.Label(left, text="ENGINE 1", bg="#0b132b", fg="#5bc0eb", font=("Courier", 12, "underline"))
        eng1_label.pack(anchor="w", pady=(0, 5))
        
        self.eng1_vars = {}
        eng1_fields = [("N1 (%)", "n1"), ("N2 (%)", "n2"), ("EGT (°F)", "egt"),
                       ("Fuel Flow (GPH)", "ff"), ("Igniter", "ign"), ("Starter", "start")]
        
        for label, key in eng1_fields:
            row = tk.Frame(left, bg="#0b132b")
            row.pack(fill="x", pady=2)
            
            lbl = tk.Label(row, text=label, bg="#0b132b", fg="#8e9aaf", font=("Courier", 11), width=15, anchor="w")
            lbl.pack(side="left")
            
            var = tk.StringVar(value="---")
            self.eng1_vars[key] = var
            
            val_lbl = tk.Label(row, textvariable=var, bg="#0b132b", fg="white", font=("Courier", 12, "bold"))
            val_lbl.pack(side="right")

        # Spacer
        spacer1 = tk.Label(left, bg="#0b132b")
        spacer1.pack(pady=10)

        # Engine 2 block
        eng2_label = tk.Label(left, text="ENGINE 2", bg="#0b132b", fg="#5bc0eb", font=("Courier", 12, "underline"))
        eng2_label.pack(anchor="w", pady=(0, 5))
        
        self.eng2_vars = {}
        eng2_fields = [("N1 (%)", "n1"), ("N2 (%)", "n2"), ("EGT (°F)", "egt"),
                       ("Fuel Flow (GPH)", "ff"), ("Igniter", "ign"), ("Starter", "start")]
                       
        for label, key in eng2_fields:
            row = tk.Frame(left, bg="#0b132b")
            row.pack(fill="x", pady=2)
            
            lbl = tk.Label(row, text=label, bg="#0b132b", fg="#8e9aaf", font=("Courier", 11), width=15, anchor="w")
            lbl.pack(side="left")
            
            var = tk.StringVar(value="---")
            self.eng2_vars[key] = var
            
            val_lbl = tk.Label(row, textvariable=var, bg="#0b132b", fg="white", font=("Courier", 12, "bold"))
            val_lbl.pack(side="right")

        # Spacer
        spacer2 = tk.Label(left, bg="#0b132b")
        spacer2.pack(pady=10)

        # Flight Data block
        flight_label = tk.Label(left, text="FLIGHT CONDITIONS", bg="#0b132b", fg="#5bc0eb", font=("Courier", 12, "underline"))
        flight_label.pack(anchor="w", pady=(0, 5))
        
        self.flight_vars = {}
        flight_fields = [("Altitude (ft)", "alt"), ("Airspeed (kts)", "spd"),
                         ("OAT (°F)", "oat"), ("Static Press (inHg)", "press")]
                         
        for label, key in flight_fields:
            row = tk.Frame(left, bg="#0b132b")
            row.pack(fill="x", pady=2)
            
            lbl = tk.Label(row, text=label, bg="#0b132b", fg="#8e9aaf", font=("Courier", 11), width=15, anchor="w")
            lbl.pack(side="left")
            
            var = tk.StringVar(value="---")
            self.flight_vars[key] = var
            
            val_lbl = tk.Label(row, textvariable=var, bg="#0b132b", fg="white", font=("Courier", 12, "bold"))
            val_lbl.pack(side="right")

        # ---------- Middle Panel: ML Prognostics ----------
        middle = tk.Frame(main, bg="#1c2541", width=500)
        middle.pack(side="left", fill="both", expand=True, padx=(0, 20))

        # Big status label
        self.status_var = tk.StringVar(value="WAITING FOR DATA")
        self.status_label = tk.Label(middle, textvariable=self.status_var,
                                      bg="#1c2541", fg="white",
                                      font=("Helvetica", 36, "bold"), anchor="w")
        self.status_label.pack(fill="x", padx=30, pady=(20, 10))

        # RUL and confidence boxes
        metrics = tk.Frame(middle, bg="#1c2541")
        metrics.pack(fill="x", padx=30, pady=10)

        # RUL Frame
        rul_frame = tk.Frame(metrics, bg="#0b132b", padx=20, pady=15)
        rul_frame.pack(side="left", expand=True, fill="x", padx=(0, 10))
        
        self.rul_var = tk.StringVar(value="--")
        rul_val_lbl = tk.Label(rul_frame, textvariable=self.rul_var, bg="#0b132b", fg="#5bc0eb", font=("Helvetica", 32, "bold"))
        rul_val_lbl.pack()
        
        rul_text_lbl = tk.Label(rul_frame, text="EST. RUL (CYCLES)", bg="#0b132b", fg="#8e9aaf", font=("Courier", 10))
        rul_text_lbl.pack()

        # Confidence Frame
        conf_frame = tk.Frame(metrics, bg="#0b132b", padx=20, pady=15)
        conf_frame.pack(side="right", expand=True, fill="x", padx=(10, 0))
        
        self.conf_var = tk.StringVar(value="--%")
        conf_val_lbl = tk.Label(conf_frame, textvariable=self.conf_var, bg="#0b132b", fg="#5bc0eb", font=("Helvetica", 32, "bold"))
        conf_val_lbl.pack()
        
        conf_text_lbl = tk.Label(conf_frame, text="AI CONFIDENCE", bg="#0b132b", fg="#8e9aaf", font=("Courier", 10))
        conf_text_lbl.pack()

        # AI explanation box
        exp_lbl = tk.Label(middle, text="AI EXPLANATION & AMM REFERENCE", bg="#1c2541", fg="#8e9aaf", font=("Courier", 11, "underline"), anchor="w")
        exp_lbl.pack(fill="x", padx=30, pady=(20, 5))
        
        self.action_var = tk.StringVar(value="Connecting to FlightGear...")
        self.action_label = tk.Label(middle, textvariable=self.action_var, bg="#1c2541", fg="white", font=("Courier", 10), justify="left", anchor="nw", wraplength=450)
        self.action_label.pack(fill="both", expand=True, padx=30, pady=(0, 20))

        # ---------- Right Panel: GOD MODE CONTROLS ----------
        right = tk.Frame(main, bg="#2d1b3d", width=350)
        right.pack(side="right", fill="y")

        god_mode_lbl = tk.Label(right, text="🔥 GOD MODE OVERRIDES", bg="#2d1b3d", fg="#ff006e", font=("Courier", 14, "bold"))
        god_mode_lbl.pack(pady=(20, 10))

        # Active fault indicator
        self.fault_indicator_var = tk.StringVar(value="NO FAULT ACTIVE")
        self.fault_indicator = tk.Label(right, textvariable=self.fault_indicator_var,
                                        bg="#2d1b3d", fg="#00ff41",
                                        font=("Courier", 11, "bold"), wraplength=250)
        self.fault_indicator.pack(pady=10)

        # ==========================================
        # GROUND FAULT BUTTONS
        # ==========================================
        ground_lbl = tk.Label(right, text="🛬 GROUND FAULTS:", bg="#2d1b3d", fg="#8e9aaf", font=("Courier", 10, "bold"))
        ground_lbl.pack(anchor="w", padx=20, pady=(10, 5))

        self.btn_hot_start = tk.Button(right, text="Hot Start", command=lambda: self.inject_fault('hot_start', 'ground'), bg="#e63946", fg="white", width=25, state="disabled")
        self.btn_hot_start.pack(pady=2)

        self.btn_hung_start = tk.Button(right, text="Hung Start", command=lambda: self.inject_fault('hung_start', 'ground'), bg="#f4a261", fg="white", width=25, state="disabled")
        self.btn_hung_start.pack(pady=2)

        self.btn_misfire = tk.Button(right, text="Misfire", command=lambda: self.inject_fault('misfire', 'ground'), bg="#e76f51", fg="white", width=25, state="disabled")
        self.btn_misfire.pack(pady=2)

        self.btn_weak_spark = tk.Button(right, text="Weak Spark", command=lambda: self.inject_fault('weak_spark', 'ground'), bg="#e76f51", fg="white", width=25, state="disabled")
        self.btn_weak_spark.pack(pady=2)

        self.btn_exciter = tk.Button(right, text="Exciter Box Failure", command=lambda: self.inject_fault('exciter_box_failure', 'ground'), bg="#bc4749", fg="white", width=25, state="disabled")
        self.btn_exciter.pack(pady=2)

        self.btn_fouling = tk.Button(right, text="Igniter Fouling", command=lambda: self.inject_fault('igniter_fouling', 'ground'), bg="#f4a261", fg="white", width=25, state="disabled")
        self.btn_fouling.pack(pady=2)

        # ==========================================
        # AIRBORNE FAULT BUTTONS
        # ==========================================
        airborne_lbl = tk.Label(right, text="✈️ AIRBORNE FAULTS:", bg="#2d1b3d", fg="#8e9aaf", font=("Courier", 10, "bold"))
        airborne_lbl.pack(anchor="w", padx=20, pady=(15, 5))

        self.btn_flameout = tk.Button(right, text="In-Flight Flameout", command=lambda: self.inject_fault('engine_flameout', 'airborne'), bg="#bc4749", fg="white", width=25, state="disabled")
        self.btn_flameout.pack(pady=2)

        self.btn_degradation = tk.Button(right, text="Igniter Degradation", command=lambda: self.inject_fault('igniter_degradation', 'airborne'), bg="#6a4c93", fg="white", width=25, state="disabled")
        self.btn_degradation.pack(pady=2)

        self.btn_erosion = tk.Button(right, text="Electrode Erosion", command=lambda: self.inject_fault('electrode_erosion', 'airborne'), bg="#6a4c93", fg="white", width=25, state="disabled")
        self.btn_erosion.pack(pady=2)

        self.btn_insulation = tk.Button(right, text="Insulation Breakdown", command=lambda: self.inject_fault('insulation_breakdown', 'airborne'), bg="#bc4749", fg="white", width=25, state="disabled")
        self.btn_insulation.pack(pady=2)

        self.btn_relight = tk.Button(right, text="High-Alt Relight Fail", command=lambda: self.inject_fault('high_altitude_relight', 'airborne'), bg="#bc4749", fg="white", width=25, state="disabled")
        self.btn_relight.pack(pady=2)

        # ==========================================
        # CLEAR FAULT BUTTON
        # ==========================================
        self.btn_clear = tk.Button(right, text="CLEAR ALL FAULTS", command=self.clear_fault, bg="#00ff41", fg="black", font=("Courier", 11, "bold"), width=20)
        self.btn_clear.pack(pady=20)

    # ---------------------------- Flight Phase Detection ----------------------------
    def detect_flight_phase(self, fg):
        alt = fg['altitude_ft']
        spd = fg['airspeed_kts']
        n1 = fg['eng1_n1']

        if alt < 500 and spd < 30:
            return 'ground'
        elif alt < 1500 and n1 > 70:
            return 'takeoff'
        elif alt > 25000:
            return 'cruise'
        elif alt > 1000:
            return 'airborne'
        else:
            return 'ground'

    # ---------------------------- Context-Aware Test Suggestions ----------------------------
    def offer_tests(self, phase):
        if phase == self.last_flight_phase or self.test_offer_shown:
            return

        self.last_flight_phase = phase
        self.test_offer_shown = True

        if phase == 'ground':
            self.btn_hot_start.config(state="normal")
            self.btn_hung_start.config(state="normal")
            self.btn_misfire.config(state="normal")
            self.btn_weak_spark.config(state="normal")
            self.btn_exciter.config(state="normal")
            self.btn_fouling.config(state="normal")
            
            self.btn_flameout.config(state="disabled")
            self.btn_degradation.config(state="disabled")
            self.btn_erosion.config(state="disabled")
            self.btn_insulation.config(state="disabled")
            self.btn_relight.config(state="disabled")

            response = messagebox.askyesno(
                "Ground Test Available",
                "🛬 AIRCRAFT ON GROUND DETECTED\n\n"
                "Would you like to test ignition fault scenarios?\n\n"
                "These tests are SAFE to run on the ground."
            )

            if not response:
                self.test_offer_shown = False

        elif phase in ['airborne', 'cruise']:
            self.btn_hot_start.config(state="disabled")
            self.btn_hung_start.config(state="disabled")
            self.btn_misfire.config(state="disabled")
            self.btn_weak_spark.config(state="disabled")
            self.btn_exciter.config(state="disabled")
            self.btn_fouling.config(state="disabled")
            
            self.btn_flameout.config(state="normal")
            self.btn_degradation.config(state="normal")
            self.btn_erosion.config(state="normal")
            self.btn_insulation.config(state="normal")
            self.btn_relight.config(state="normal")

            response = messagebox.askyesno(
                "Airborne Test Available",
                "✈️ AIRCRAFT AIRBORNE DETECTED\n\n"
                "Would you like to test in-flight fault scenarios?\n\n"
                "⚠️ WARNING: These simulate critical failures!"
            )

            if not response:
                self.test_offer_shown = False

    # ---------------------------- Fault Injection ----------------------------
    def inject_fault(self, fault_name, category):
        if fault_name not in FAULT_SCENARIOS[category]:
            return

        fault = FAULT_SCENARIOS[category][fault_name]

        confirm = messagebox.askyesno(
            f"Confirm Fault Injection",
            f"⚠️ INJECT FAULT: {fault['name']}\n\n"
            f"Description: {fault['description']}\n"
            f"Risk Level: {fault['risk']}\n\n"
            f"This will override physics parameters.\n"
            f"AI will respond based on injected fault.\n\n"
            f"Proceed?"
        )

        if confirm:
            with self.lock:
                self.active_fault = fault_name
                self.fault_overrides = fault['modifications']
            self.fault_indicator_var.set(f"ACTIVE: {fault['name']}")
            self.fault_indicator.config(fg="#ff006e")

    def clear_fault(self):
        with self.lock:
            self.active_fault = None
            self.fault_overrides = {}
        self.fault_indicator_var.set("NO FAULT ACTIVE")
        self.fault_indicator.config(fg="#00ff41")
        self.test_offer_shown = False

    # ---------------------------- UDP Listener ----------------------------
    def parse_flightgear_data(self, data_str):
        parts = data_str.strip().split(',')
        if len(parts) < 21:
            return None
        try:
            return {
                'timestamp': float(parts[0]),
                'eng1_n1': float(parts[1]), 'eng1_n2': float(parts[2]),
                'eng1_egt': float(parts[3]), 'eng1_ff': float(parts[4]),
                'eng1_oil_temp_degf': float(parts[5]), 'eng1_oil_pressure_psi': float(parts[6]),
                'eng1_thrust_lbs': float(parts[7]),
                'eng1_ign': int(parts[8]), 'eng1_start': int(parts[9]), 'eng1_cutoff': int(parts[10]),
                'eng2_n1': float(parts[11]), 'eng2_n2': float(parts[12]),
                'eng2_egt': float(parts[13]), 'eng2_ff': float(parts[14]),
                'eng2_ign': int(parts[15]), 'eng2_start': int(parts[16]),
                'altitude_ft': float(parts[17]), 'airspeed_kts': float(parts[18]),
                'oat_degf': float(parts[19]), 'static_pressure_inhg': float(parts[20]),
            }
        except ValueError:
            return None

    def engineer_features(self, fg, egt_rise_rate):
        """Convert raw/faulted FlightGear telemetry into the 19 ML features."""
        is_igniting = fg['eng1_ign'] == 1
        
        # FlightGear Autostart Compensation: 
        # If the engine is running (N2 > 10%) and fuel is flowing, force igniter True for AI logic.
        if fg['eng1_ff'] > 10 and fg['eng1_n2'] > 10 and not self.active_fault:
            is_igniting = True
        
        # 1. Calculate Baseline Features (Healthy Engine)
        spark_energy = np.random.normal(3.5, 0.2) if is_igniting else 0.0
        spark_volt = np.random.normal(2000, 50) if is_igniting else 0.0
        spark_curr = np.random.normal(18, 1) if is_igniting else 0.0
        spark_std = np.random.normal(45, 5) if is_igniting else 0.0
        vib_amp = np.random.normal(150, 20) + (fg['eng1_n1'] * 2.5)
        vib_rms = vib_amp * 0.707
        resistance = np.random.normal(1.5, 0.05)
        decay_rate = np.random.normal(1200, 50) if is_igniting else 0.0
        combustion_eff = 0.98 if fg['eng1_n1'] > 20 else 0.0

        # 2. DEEP FEATURE SPOOFING (Forces the AI to see low RUL)
        with self.lock:
            current_fault = self.active_fault

        # >>> ADDED HOT START AND HUNG START DEEP SPOOFING HERE <<<
        if current_fault == 'hot_start':
            resistance = np.random.normal(80.0, 5.0)   # High wear forces low RUL
            decay_rate = np.random.normal(300, 20)
            
        elif current_fault == 'hung_start':
            resistance = np.random.normal(85.0, 5.0)   # High wear forces low RUL
            decay_rate = np.random.normal(300, 20)
            
        elif current_fault == 'exciter_box_failure':
            spark_energy = np.random.normal(0.5, 0.1)  # Weak joules
            spark_volt = np.random.normal(800, 50)     # Low voltage
            decay_rate = np.random.normal(300, 20)     # Bad decay
            resistance = np.random.normal(50.0, 5.0)   # High resistance = Low RUL
            
        elif current_fault == 'electrode_erosion':
            spark_energy = np.random.normal(1.2, 0.2)
            resistance = np.random.normal(120.0, 10.0) # Extreme resistance = Critical RUL
            vib_amp *= 1.8
            vib_rms *= 1.8
            
        elif current_fault == 'igniter_fouling':
            resistance = np.random.normal(80.0, 5.0)   # Carbon buildup
            spark_std = np.random.normal(150, 20)      # Erratic spark
            
        elif current_fault == 'insulation_breakdown':
            spark_volt = np.random.normal(500, 200)    # Voltage leaking
            spark_std = np.random.normal(300, 50)      # Massive variance
            resistance = np.random.normal(0.1, 0.01)   # Short circuit
            
        elif current_fault in ['misfire', 'engine_flameout']:
            spark_energy = 0.0
            spark_volt = 0.0
            spark_curr = 0.0
            combustion_eff = 0.0
            resistance = np.random.normal(150.0, 5.0)  # Extreme wear forces 0-5 RUL
            
        elif current_fault == 'igniter_degradation':
            spark_energy = np.random.normal(2.1, 0.2)
            resistance = np.random.normal(35.0, 3.0)
            
        elif current_fault == 'weak_spark':
            spark_energy = np.random.normal(1.2, 0.1)
            resistance = np.random.normal(20.0, 2.0)
            
        elif current_fault == 'high_altitude_relight':
            spark_energy = np.random.normal(1.8, 0.2)
            resistance = np.random.normal(25.0, 2.0)

        # 3. Assemble the 19-Feature Payload
        return {
            'spark_energy_J': spark_energy,
            'spark_peak_voltage_V': spark_volt,
            'spark_peak_current_A': spark_curr,
            'spark_voltage_std_V': spark_std,
            'ignition_delay_s': 0.0,
            'ignition_success': 1 if fg['eng1_ff'] > 100 and egt_rise_rate > 5 else 0,
            'max_EGT_rise_rate_Kps': max(0.0, egt_rise_rate),
            'time_to_stable_s': 0.0,
            'peak_chamber_pressure_bar': np.random.normal(30, 2) * (fg['eng1_n2']/100 + 0.1),
            'mean_chamber_pressure_bar': np.random.normal(25, 1.5) * (fg['eng1_n2']/100 + 0.1),
            'vibration_peak_amplitude_Pa': vib_amp,
            'vibration_rms_Pa': vib_rms,
            'vibration_dominant_freq_Hz': fg['eng1_n1'] * 1.5,
            'EGT_std_K': np.std(self.egt_history) if len(self.egt_history) > 1 else 0.0,
            'pressure_std_bar': np.random.normal(1.2, 0.1),
            'combustion_efficiency': combustion_eff,
            'spark_efficiency': 0.95 if is_igniting else 0.0,
            'igniter_resistance_ohm': resistance,
            'spark_decay_rate': decay_rate
        }

    def udp_listener(self):
        counter = 0
        while self.running:
            try:
                data, _ = self.sock.recvfrom(BUFFER_SIZE)
                fg = self.parse_flightgear_data(data.decode('utf-8'))
                if not fg:
                    continue

                # Calculate Rates (dt)
                egt_k = (fg['eng1_egt'] - 32) * 5.0/9.0 + 273.15
                self.egt_history.append(egt_k)
                
                dt = fg['timestamp'] - self.prev_time if self.prev_time else 0.1
                if dt <= 0: dt = 0.1
                egt_rise_rate = (egt_k - self.prev_egt_k) / dt if self.prev_egt_k else 0.0
                
                self.prev_time = fg['timestamp']
                self.prev_egt_k = egt_k

                # ==========================================
                # ⚠️ PRIMARY PHYSICS SPOOFING (Visuals)
                # ==========================================
                with self.lock:
                    current_fault = self.active_fault
                
                if current_fault == "hot_start":
                    fg['eng1_ign'], fg['eng2_ign'] = 1, 1          
                    fg['eng1_start'], fg['eng2_start'] = 1, 1        
                    fg['eng1_ff'], fg['eng2_ff'] = 5000.0, 5000.0   # Matches Dataset max limit
                    egt_rise_rate = 450.0                           # Matches Dataset max limit
                    fg['eng1_egt'], fg['eng2_egt'] = 950.0, 950.0 
                    
                elif current_fault == "hung_start":
                    fg['eng1_ign'], fg['eng2_ign'] = 1, 1          
                    fg['eng1_start'], fg['eng2_start'] = 1, 1
                    fg['eng1_n2'], fg['eng2_n2'] = 15.0, 15.0 
                    fg['eng1_n1'], fg['eng2_n1'] = 2.0, 2.0
                    egt_rise_rate = 5.0
                    
                elif current_fault == "misfire":
                    fg['eng1_ign'], fg['eng2_ign'] = 0, 0
                    fg['eng1_ff'], fg['eng2_ff'] = 100.0, 100.0
                    egt_rise_rate = -50.0                           # Matches Dataset misfire behavior
                    
                elif current_fault == "engine_flameout":
                    fg['eng1_ign'], fg['eng2_ign'] = 0, 0          
                    fg['eng1_ff'], fg['eng2_ff'] = 0.0, 0.0
                    egt_rise_rate = -50.0
                    fg['eng1_egt'], fg['eng2_egt'] = 400.0, 400.0
                    
                elif current_fault == "igniter_degradation":
                    fg['eng1_ign'], fg['eng2_ign'] = 1, 1
                    egt_rise_rate = 15.0
                    
                elif current_fault == "exciter_box_failure":
                    fg['eng1_ign'], fg['eng2_ign'] = 1, 1
                    egt_rise_rate = 10.0
                    
                elif current_fault == "igniter_fouling":
                    fg['eng1_ign'], fg['eng2_ign'] = 1, 1
                    egt_rise_rate = 8.0
                    
                elif current_fault == "electrode_erosion":
                    fg['eng1_ign'], fg['eng2_ign'] = 1, 1
                    egt_rise_rate = 12.0
                    
                elif current_fault == "insulation_breakdown":
                    fg['eng1_ign'], fg['eng2_ign'] = 1, 1
                    egt_rise_rate = 5.0
                    
                elif current_fault == "weak_spark":
                    fg['eng1_ign'], fg['eng2_ign'] = 1, 1
                    egt_rise_rate = 15.0
                    
                elif current_fault == "high_altitude_relight":
                    fg['eng1_ign'], fg['eng2_ign'] = 1, 1
                    egt_rise_rate = 8.0
                # ==========================================

                # Store latest telemetry (thread-safe)
                with self.lock:
                    self.latest_fg = fg

                # Detect flight phase and offer tests
                phase = self.detect_flight_phase(fg)
                self.offer_tests(phase)

                # Run inference every 10 packets (~1 second)
                counter += 1
                if counter % 10 == 0:
                    features = self.engineer_features(fg, egt_rise_rate)
                    result = self.predictor.predict(features)
                    with self.lock:
                        self.latest_result = result
            except Exception:
                pass

    # ---------------------------- GUI Update ----------------------------
    def update_gui(self):
        with self.lock:
            fg = self.latest_fg
            res = self.latest_result
            current_fault = self.active_fault

        if fg:
            self.eng1_vars['n1'].set(f"{fg['eng1_n1']:.1f}")
            self.eng1_vars['n2'].set(f"{fg['eng1_n2']:.1f}")
            self.eng1_vars['egt'].set(f"{fg['eng1_egt']:.1f}")
            self.eng1_vars['ff'].set(f"{fg['eng1_ff']:.1f}")
            self.eng1_vars['ign'].set("ON" if fg['eng1_ign'] else "OFF")
            self.eng1_vars['start'].set("ON" if fg['eng1_start'] else "OFF")

            self.eng2_vars['n1'].set(f"{fg['eng2_n1']:.1f}")
            self.eng2_vars['n2'].set(f"{fg['eng2_n2']:.1f}")
            self.eng2_vars['egt'].set(f"{fg['eng2_egt']:.1f}")
            self.eng2_vars['ff'].set(f"{fg['eng2_ff']:.1f}")
            self.eng2_vars['ign'].set("ON" if fg['eng2_ign'] else "OFF")
            self.eng2_vars['start'].set("ON" if fg['eng2_start'] else "OFF")

            self.flight_vars['alt'].set(f"{fg['altitude_ft']:.0f}")
            self.flight_vars['spd'].set(f"{fg['airspeed_kts']:.1f}")
            self.flight_vars['oat'].set(f"{fg['oat_degf']:.1f}")
            self.flight_vars['press'].set(f"{fg['static_pressure_inhg']:.2f}")

        if res and res.get('status') == 'success':
            status = res['health_status'].upper()
            
            if status == "HEALTHY":
                colour = "#2a9d8f"
            elif status == "MONITOR":
                colour = "#f4a261"
            elif status in ("CAUTION", "CRITICAL", "FAILURE"):
                colour = "#e63946"
            else:
                colour = "white"

            self.status_var.set(status)
            self.status_label.config(fg=colour)
            self.rul_var.set(f"{res['predicted_rul_cycles']:.0f}")
            self.conf_var.set(f"{res['confidence']*100:.1f}%")

            if fg:
                alt = fg['altitude_ft']
                spd = fg['airspeed_kts']
                
                if alt < 500 and spd < 30:
                    phase = "On Ground / Taxi"
                elif alt < 1500 and spd > 50:
                    phase = "Takeoff"
                elif alt < 10000:
                    phase = "Climb"
                elif alt > 25000:
                    phase = "Cruise"
                else:
                    phase = "Descent / Approach"

                explanation = (
                    f"• Flight Phase: {phase}\n"
                    f"• Altitude: {alt:.0f} ft   Airspeed: {spd:.1f} kts\n"
                    f"• OAT: {fg['oat_degf']:.1f} °F   Static Pressure: {fg['static_pressure_inhg']:.2f} inHg\n"
                    f"\n"
                )

                if current_fault:
                    explanation += f"🔥 ACTIVE FAULT: {current_fault.upper().replace('_', ' ')}\n\n"

                explanation += (
                    f"• Ignition System Status: {status}\n"
                    f"• Est. Remaining Useful Life: {res['predicted_rul_cycles']:.0f} cycles\n"
                    f"• AI Confidence: {res['confidence']*100:.1f}%\n"
                    f"\n"
                    f"• CS-25 / AMM Reference:\n"
                    f"  {res['action']}\n"
                )
                
                if fg['eng1_ign']:
                    explanation += "\n• Igniter is ON (continuous ignition mode)."
                if fg['eng1_start']:
                    explanation += "\n• Starter is engaged."

                self.action_var.set(explanation)
            else:
                self.action_var.set(res['action'])
        else:
            pass

        self.root.after(100, self.update_gui)

    def stop(self):
        self.running = False
        self.sock.close()

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitalTwinGUI(root)

    def on_closing():
        app.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()