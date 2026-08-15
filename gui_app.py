"""
Industrial Quality Control - Desktop HMI & Management Application
===================================================================
A native Windows PyQt6 GUI application to manage image classification,
Modbus TCP (Server & Client modes), webcam settings, model training,
and real-time inspection HMI screen with 5-second auto-reset.

Features:
  1. HMI Dashboard: Live status, inspection result display, 5s countdown timer,
     and production KPI counters (Total, Good, Bad, Yield %).
  2. Modbus TCP Engine: Configurable for SERVER mode or CLIENT mode with custom
     coil addresses (Trigger, Result, Busy) and manual coil testing.
  3. Camera & Model Manager: Webcam index selection, preview test, and
     in-app model training with real-time log output.
  4. Inspection History: Table of historical triggers and image preview.

Usage:
    python gui_app.py
"""

import os
import sys
import time
import threading
from datetime import datetime

class DummyStream:
    def write(self, text): pass
    def flush(self): pass

if sys.stdout is None:
    sys.stdout = DummyStream()
if sys.stderr is None:
    sys.stderr = DummyStream()
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor, QIcon
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QSpinBox, QComboBox, QCheckBox,
    QTabWidget, QGroupBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QTextEdit, QProgressBar, QFrame, QSplitter, QFileDialog, QMessageBox
)

from pymodbus.client import ModbusTcpClient
from pymodbus.server import StartTcpServer
from pymodbus.datastore import (
    ModbusSlaveContext, ModbusServerContext, ModbusSequentialDataBlock
)
from pymodbus.device import ModbusDeviceIdentification


# ──────────────────────────────────────────────
# Config & Paths
# ──────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(SCRIPT_DIR, "model.pth")
DEFAULT_DATA_DIR = os.path.join(SCRIPT_DIR, "dataset")
DEFAULT_PRODUCTION_DIR = os.path.join(SCRIPT_DIR, "production")
INPUT_SIZE = 256


# ──────────────────────────────────────────────
# Image Preprocessing & Hybrid Model (same as train.py)
# ──────────────────────────────────────────────
def extract_object_roi(pil_img: Image.Image) -> Image.Image:
    """Isolates the object from background tables, floor, or environment."""
    img_np = np.array(pil_img)
    h, w = img_np.shape[:2]

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area > (h * w * 0.15):
            x, y, bw, bh = cv2.boundingRect(c)
            pad = 10
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(w, x + bw + pad), min(h, y + bh + pad)
            cropped = img_np[y1:y2, x1:x2]
            return Image.fromarray(cropped)

    crop_h, crop_w = int(h * 0.1), int(w * 0.1)
    cropped = img_np[crop_h:h - crop_h, crop_w:w - crop_w]
    return Image.fromarray(cropped)


def get_color_and_text_stats(pil_img: Image.Image) -> torch.Tensor:
    """Extracts 13 explicit LAB/HSV color, shade, and text sharpness metrics."""
    img_np = np.array(pil_img)

    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    l_mean, l_std = float(lab[:, :, 0].mean()), float(lab[:, :, 0].std())
    a_mean, a_std = float(lab[:, :, 1].mean()), float(lab[:, :, 1].std())
    b_mean, b_std = float(lab[:, :, 2].mean()), float(lab[:, :, 2].std())

    h_mean, h_std = float(hsv[:, :, 0].mean()), float(hsv[:, :, 0].std())
    s_mean, s_std = float(hsv[:, :, 1].mean()), float(hsv[:, :, 1].std())
    v_mean, v_std = float(hsv[:, :, 2].mean()), float(hsv[:, :, 2].std())

    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    stats = np.array([
        l_mean / 255.0, l_std / 255.0,
        a_mean / 255.0, a_std / 255.0,
        b_mean / 255.0, b_std / 255.0,
        h_mean / 179.0, h_std / 179.0,
        s_mean / 255.0, s_std / 255.0,
        v_mean / 255.0, v_std / 255.0,
        float(np.log1p(laplacian_var) / 10.0)
    ], dtype=np.float32)

    return torch.from_numpy(stats)


class HybridClassifier(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.resnet = models.resnet18(weights=None)
        self.resnet.fc = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(512 + 13, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, color_stats):
        resnet_feats = self.resnet(x)
        combined = torch.cat([resnet_feats, color_stats], dim=1)
        return self.classifier(combined)


predict_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ──────────────────────────────────────────────
# Classifier Inference Helper
# ──────────────────────────────────────────────
class ClassifierEngine:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.idx_to_label = {0: "good", 1: "bad"}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model(model_path)

    def load_model(self, model_path: str):
        if not os.path.isfile(model_path):
            print(f"[-] Model file not found: {model_path}")
            return False

        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            label_map = checkpoint.get("label_map", {"good": 0, "bad": 1})
            self.idx_to_label = {v: k for k, v in label_map.items()}

            self.model = HybridClassifier(num_classes=len(label_map))
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()
            self.model_path = model_path
            print(f"[+] Loaded classifier model: {model_path}")
            return True
        except Exception as e:
            print(f"[-] Failed to load model: {e}")
            return False

    def classify_pil(self, pil_image: Image.Image):
        if self.model is None:
            return "unknown", 0.0, [0.5, 0.5]

        roi_image = extract_object_roi(pil_image)
        color_stats = get_color_and_text_stats(roi_image).unsqueeze(0).to(self.device)
        spatial_tensor = predict_transform(roi_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(spatial_tensor, color_stats)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred_idx = probs.max(1)

        label = self.idx_to_label.get(pred_idx.item(), "good")
        conf = confidence.item() * 100
        return label, conf, probs[0].tolist()


# ──────────────────────────────────────────────
# Modbus & Worker Thread
# ──────────────────────────────────────────────
class ModbusWorker(QThread):
    # Signals to GUI
    trigger_detected = pyqtSignal(str)              # timestamp
    inspection_completed = pyqtSignal(dict)         # result dict
    status_changed = pyqtSignal(str, str)           # (status_text, style_class)
    modbus_log_signal = pyqtSignal(str)             # log message
    coil_state_signal = pyqtSignal(bool, bool, bool, bool)# (trigger, good, bad, busy)
    watch_data_signal = pyqtSignal(dict)            # live register watch data
    connection_info_signal = pyqtSignal(dict)       # connection details

    def __init__(self, config: dict, classifier: ClassifierEngine):
        super().__init__()
        self.config = config
        self.classifier = classifier
        self.running = False
        self.server_instance = None
        self.client_instance = None
        self.server_context = None
        self._poll_counter = 0
        self.reset_coils_flag = False

    def request_reset_all_coils(self):
        """Flag to zero out all Modbus coils and re-arm rising edge detection."""
        self.reset_coils_flag = True

    def update_config(self, new_config: dict):
        self.config = new_config

    def run(self):
        self.running = True
        mode = self.config.get("mode", "Server")
        host = self.config.get("host", "127.0.0.1")
        port = int(self.config.get("port", 5020))
        unit_id = int(self.config.get("unit_id", 1))
        trigger_coil = int(self.config.get("trigger_coil", 1))
        good_coil = int(self.config.get("good_coil", 10))
        bad_coil = int(self.config.get("bad_coil", 12))
        busy_coil = int(self.config.get("busy_coil", 11))
        poll_interval = float(self.config.get("poll_interval_ms", 50)) / 1000.0
        camera_idx = int(self.config.get("camera_idx", 0))

        self.modbus_log_signal.emit(f"[+] Starting Modbus engine in {mode.upper()} mode...")

        if mode == "Server":
            self._run_server(host, port, trigger_coil, good_coil, bad_coil, busy_coil, camera_idx, poll_interval)
        else:
            self._run_client(host, port, unit_id, trigger_coil, good_coil, bad_coil, busy_coil, camera_idx, poll_interval)

    def _run_server(self, host, port, trigger_coil, good_coil, bad_coil, busy_coil, camera_idx, poll_interval):
        max_coil = max(trigger_coil, good_coil, bad_coil, busy_coil) + 10
        store = ModbusSlaveContext(
            co=ModbusSequentialDataBlock(0, [0] * (max_coil + 1)),
            di=ModbusSequentialDataBlock(0, [0] * (max_coil + 1)),
            hr=ModbusSequentialDataBlock(0, [0] * (max_coil + 1)),
            ir=ModbusSequentialDataBlock(0, [0] * (max_coil + 1)),
        )
        self.server_context = ModbusServerContext(slaves=store, single=True)

        identity = ModbusDeviceIdentification()
        identity.VendorName = "ImageClassifier"
        identity.ProductName = "Quality Inspection Server"

        # Launch TCP Server in daemon thread
        server_thread = threading.Thread(
            target=StartTcpServer,
            kwargs={
                "context": self.server_context,
                "identity": identity,
                "address": (host, port)
            },
            daemon=True
        )
        server_thread.start()
        self.status_changed.emit(f"SERVER LISTENING ({host}:{port}) - Waiting for trigger...", "listening")
        self.modbus_log_signal.emit(f"[+] Modbus TCP Server running on {host}:{port}")

        self.connection_info_signal.emit({
            "mode": "SERVER", "host": host, "port": port,
            "status": "Listening", "unit_id": "-",
        })

        prev_trigger = False

        while self.running:
            try:
                slave = self.server_context[0x00]

                # Check if system returned to IDLE mode -> Reset all coils to 0
                if self.reset_coils_flag:
                    slave.setValues(1, trigger_coil, [0])
                    slave.setValues(1, good_coil, [0])
                    slave.setValues(1, bad_coil, [0])
                    slave.setValues(1, busy_coil, [0])
                    prev_trigger = False
                    self.reset_coils_flag = False
                    self.status_changed.emit(f"IDLE — SERVER LISTENING ({host}:{port})", "listening")
                    self.modbus_log_signal.emit("[+] System returned to IDLE — All Modbus registers reset to 0 (OFF). Ready for next trigger.")

                trig_val = bool(slave.getValues(1, trigger_coil, count=1)[0])
                good_val = bool(slave.getValues(1, good_coil, count=1)[0])
                bad_val = bool(slave.getValues(1, bad_coil, count=1)[0])
                busy_val = bool(slave.getValues(1, busy_coil, count=1)[0])

                self.coil_state_signal.emit(trig_val, good_val, bad_val, busy_val)

                # Emit watch data every 10 polls (~500ms at 50ms interval)
                self._poll_counter += 1
                if self._poll_counter % 10 == 0:
                    watch = {}
                    max_addr = max(trigger_coil, good_coil, bad_coil, busy_coil) + 5
                    for addr in range(max_addr + 1):
                        try:
                            v = slave.getValues(1, addr, count=1)[0]
                            watch[addr] = bool(v)
                        except Exception:
                            pass
                    self.watch_data_signal.emit(watch)

                # Detect Rising Edge
                if trig_val and not prev_trigger:
                    self._process_inspection(
                        camera_idx=camera_idx,
                        write_good_fn=lambda val: slave.setValues(1, good_coil, [int(val)]),
                        write_bad_fn=lambda val: slave.setValues(1, bad_coil, [int(val)]),
                        write_busy_fn=lambda val: slave.setValues(1, busy_coil, [int(val)]),
                        reset_trig_fn=lambda: slave.setValues(1, trigger_coil, [0]),
                        trigger_source="Modbus Server Coil Trigger"
                    )

                prev_trigger = trig_val
                time.sleep(poll_interval)

            except Exception as e:
                self.modbus_log_signal.emit(f"[!] Server monitor loop error: {e}")
                time.sleep(1.0)

    def _run_client(self, host, port, unit_id, trigger_coil, good_coil, bad_coil, busy_coil, camera_idx, poll_interval):
        self.modbus_log_signal.emit(f"[+] Connecting to PLC Modbus Server at {host}:{port}...")
        client = ModbusTcpClient(host, port=port)

        connected = client.connect()
        if not connected:
            self.status_changed.emit(f"CLIENT ERROR - Could not connect to PLC at {host}:{port}", "error")
            self.modbus_log_signal.emit(f"[-] Failed to connect to Modbus TCP Server at {host}:{port}")
            self.connection_info_signal.emit({
                "mode": "CLIENT", "host": host, "port": port,
                "status": "DISCONNECTED", "unit_id": unit_id,
            })
            self.running = False
            return

        self.status_changed.emit(f"CLIENT CONNECTED to {host}:{port} - Polling PLC...", "connected")
        self.modbus_log_signal.emit(f"[+] Connected to PLC at {host}:{port}")
        self.connection_info_signal.emit({
            "mode": "CLIENT", "host": host, "port": port,
            "status": "Connected", "unit_id": unit_id,
        })

        prev_trigger = False

        while self.running:
            try:
                # Check if system returned to IDLE mode -> Reset all coils to 0
                if self.reset_coils_flag:
                    client.write_coil(trigger_coil, False, slave=unit_id)
                    client.write_coil(good_coil, False, slave=unit_id)
                    client.write_coil(bad_coil, False, slave=unit_id)
                    client.write_coil(busy_coil, False, slave=unit_id)
                    prev_trigger = False
                    self.reset_coils_flag = False
                    self.status_changed.emit(f"IDLE — CLIENT CONNECTED ({host}:{port})", "connected")
                    self.modbus_log_signal.emit("[+] System returned to IDLE — All Modbus registers reset to 0 (OFF). Ready for next trigger.")

                rr = client.read_coils(trigger_coil, count=1, slave=unit_id)
                if rr.isError():
                    self.modbus_log_signal.emit(f"[!] Read error on trigger coil {trigger_coil}: {rr}")
                    time.sleep(1.0)
                    continue

                trig_val = bool(rr.bits[0])

                # Read good, bad & busy coils (keyword-only args for pymodbus 3.x)
                good_rr = client.read_coils(good_coil, count=1, slave=unit_id)
                good_val = bool(good_rr.bits[0]) if not good_rr.isError() else False

                bad_rr = client.read_coils(bad_coil, count=1, slave=unit_id)
                bad_val = bool(bad_rr.bits[0]) if not bad_rr.isError() else False

                busy_rr = client.read_coils(busy_coil, count=1, slave=unit_id)
                busy_val = bool(busy_rr.bits[0]) if not busy_rr.isError() else False

                self.coil_state_signal.emit(trig_val, good_val, bad_val, busy_val)

                # Emit watch data every 10 polls
                self._poll_counter += 1
                if self._poll_counter % 10 == 0:
                    watch = {}
                    max_addr = max(trigger_coil, good_coil, bad_coil, busy_coil) + 5
                    sweep_rr = client.read_coils(0, count=max_addr + 1, slave=unit_id)
                    if not sweep_rr.isError():
                        for addr in range(min(len(sweep_rr.bits), max_addr + 1)):
                            watch[addr] = bool(sweep_rr.bits[addr])
                    self.watch_data_signal.emit(watch)

                # Rising edge trigger
                if trig_val and not prev_trigger:
                    self._process_inspection(
                        camera_idx=camera_idx,
                        write_good_fn=lambda val: client.write_coil(good_coil, bool(val), slave=unit_id),
                        write_bad_fn=lambda val: client.write_coil(bad_coil, bool(val), slave=unit_id),
                        write_busy_fn=lambda val: client.write_coil(busy_coil, bool(val), slave=unit_id),
                        reset_trig_fn=lambda: client.write_coil(trigger_coil, False, slave=unit_id),
                        trigger_source=f"Modbus PLC Client ({host}:{port})"
                    )

                prev_trigger = trig_val
                time.sleep(poll_interval)

            except Exception as e:
                self.modbus_log_signal.emit(f"[!] Client polling error: {e}")
                self.connection_info_signal.emit({
                    "mode": "CLIENT", "host": host, "port": port,
                    "status": "ERROR", "unit_id": unit_id,
                })
                time.sleep(1.0)

        client.close()
        self.connection_info_signal.emit({
            "mode": "CLIENT", "host": host, "port": port,
            "status": "Disconnected", "unit_id": unit_id,
        })

    def _process_inspection(self, camera_idx, write_good_fn, write_bad_fn, write_busy_fn, reset_trig_fn, trigger_source):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.trigger_detected.emit(ts)
        self.status_changed.emit("PROCESSING - Capturing Image & Classifying...", "processing")

        write_busy_fn(True)
        # Clear both result coils at start of inspection
        write_good_fn(False)
        write_bad_fn(False)

        try:
            # Capture photo from webcam
            cap = cv2.VideoCapture(camera_idx)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open camera index {camera_idx}")

            # Warmup
            for _ in range(15):
                cap.read()
                time.sleep(0.01)

            ret, frame_bgr = cap.read()
            cap.release()

            if not ret or frame_bgr is None:
                raise RuntimeError("Failed to capture frame from webcam")

            # Convert BGR -> RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)

            # Classify
            label, confidence, probs = self.classifier.classify_pil(pil_img)
            is_good = (label == "good")

            # Write separate Good / Bad coils
            write_good_fn(is_good)         # ON if good
            write_bad_fn(not is_good)      # ON if bad

            # Save production image
            prod_dir = Path(self.config.get("production_dir", DEFAULT_PRODUCTION_DIR))
            prod_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{label}_{confidence:.0f}pct.jpg"
            img_save_path = str(prod_dir / fname)
            cv2.imwrite(img_save_path, frame_bgr)

            result_payload = {
                "timestamp": ts,
                "label": label,
                "is_good": is_good,
                "confidence": confidence,
                "probabilities": probs,
                "image_path": img_save_path,
                "bgr_frame": frame_bgr,
                "source": trigger_source
            }

            self.inspection_completed.emit(result_payload)
            self.modbus_log_signal.emit(f"[+] Inspection Result: {label.upper()} ({confidence:.1f}%) -> Saved: {fname}")

        except Exception as e:
            self.modbus_log_signal.emit(f"[!] Error processing inspection: {e}")
            write_good_fn(False)
            write_bad_fn(True)  # Error → treat as bad
        finally:
            reset_trig_fn()
            write_busy_fn(False)

    def stop(self):
        self.running = False


# ──────────────────────────────────────────────
# Training Background Thread
# ──────────────────────────────────────────────
class TrainingThread(QThread):
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, data_dir, model_path, epochs, batch_size, lr):
        super().__init__()
        self.data_dir = data_dir
        self.model_path = model_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

    def run(self):
        from train import train_model
        try:
            self.progress_signal.emit("[+] Initializing training engine...")
            save_path = train_model(
                data_dir=self.data_dir,
                epochs=self.epochs,
                batch_size=self.batch_size,
                lr=self.lr,
                save_path=self.model_path,
                log_fn=self.progress_signal.emit
            )
            self.finished_signal.emit(True, f"Training complete! Model saved to {save_path}")
        except Exception as e:
            import traceback
            err_details = f"Training error: {e}\n\n{traceback.format_exc()}"
            self.finished_signal.emit(False, err_details)


# ──────────────────────────────────────────────
# Main Application GUI (PyQt6)
# ──────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Industrial Inspection HMI & Modbus Classifier")
        self.resize(1280, 800)
        self.setMinimumSize(1024, 700)

        # Core Engines
        self.classifier = ClassifierEngine(DEFAULT_MODEL_PATH)
        self.modbus_worker = None

        # State Variables
        self.good_count = 0
        self.bad_count = 0
        self.total_count = 0

        # Countdown Timer for 5-second Auto-Reset
        self.reset_timer = QTimer(self)
        self.reset_timer.setInterval(100)  # update 10 times/sec
        self.reset_timer.timeout.connect(self._on_reset_timer_tick)
        self.countdown_seconds = 5.0
        self.hold_time_setting = 5.0

        # Setup GUI & Apply Dark Theme
        self._apply_dark_theme()
        self._init_ui()

        # Auto-start Modbus worker
        self._restart_modbus_worker()

    # ──────────────────────────────────────────
    # UI Building
    # ──────────────────────────────────────────
    def _init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        # Header Bar
        main_layout.addWidget(self._build_header_bar())

        # Main Tabbed Area
        self.tabs = QTabWidget()
        self.tabs.setIconSize(QSize(20, 20))

        self.tab_hmi = self._build_tab_hmi()
        self.tab_modbus = self._build_tab_modbus()
        self.tab_model = self._build_tab_model()
        self.tab_history = self._build_tab_history()

        self.tabs.addTab(self.tab_hmi, "🖥  HMI Dashboard")
        self.tabs.addTab(self.tab_modbus, "⚙  Modbus & Camera Setup")
        self.tabs.addTab(self.tab_model, "🧠  Model & Training")
        self.tabs.addTab(self.tab_history, "📊  Production History")

        main_layout.addWidget(self.tabs)

    def _build_header_bar(self):
        header = QFrame()
        header.setObjectName("HeaderFrame")
        header.setFixedHeight(60)
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(15, 5, 15, 5)

        title = QLabel("INDUSTRIAL QUALITY CONTROL — AUTOMATED INSPECTION HMI")
        title.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))

        self.lbl_mode_badge = QLabel("MODE: SERVER")
        self.lbl_mode_badge.setObjectName("BadgeMode")

        self.lbl_conn_status = QLabel("STATUS: IDLE")
        self.lbl_conn_status.setObjectName("BadgeStatusIdle")

        h_layout.addWidget(title)
        h_layout.addStretch()
        h_layout.addWidget(self.lbl_mode_badge)
        h_layout.addWidget(self.lbl_conn_status)

        return header

    # ──────────────────────────────────────────
    # TAB 1: HMI DASHBOARD
    # ──────────────────────────────────────────
    def _build_tab_hmi(self):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)

        # Left Column: Image & Inspection Result Display
        left_box = QGroupBox("LIVE INSPECTION DISPLAY")
        left_layout = QVBoxLayout(left_box)

        self.lbl_image_display = QLabel()
        self.lbl_image_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_image_display.setMinimumSize(640, 480)
        self.lbl_image_display.setObjectName("ImageDisplayCard")
        self._set_idle_image_display()

        # Prominent Result Status Banner
        self.lbl_result_banner = QLabel("IDLE — WAITING FOR MODBUS TRIGGER")
        self.lbl_result_banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_result_banner.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        self.lbl_result_banner.setFixedHeight(60)
        self.lbl_result_banner.setObjectName("BannerIdle")

        # 5-Second Reset Countdown Label
        self.lbl_countdown = QLabel("")
        self.lbl_countdown.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_countdown.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.lbl_countdown.setStyleSheet("color: #8899a6;")

        left_layout.addWidget(self.lbl_image_display, stretch=1)
        left_layout.addWidget(self.lbl_result_banner)
        left_layout.addWidget(self.lbl_countdown)

        # Right Column: Production KPIs & Controls
        right_box = QGroupBox("PRODUCTION METRICS & CONTROLS")
        right_layout = QVBoxLayout(right_box)
        right_layout.setSpacing(15)

        # KPI Cards
        self.lbl_kpi_total = self._create_kpi_card("TOTAL INSPECTED", "0", "#3897f0")
        self.lbl_kpi_good = self._create_kpi_card("GOOD (PASSED)", "0", "#2ecc71")
        self.lbl_kpi_bad = self._create_kpi_card("BAD (FAILED)", "0", "#e74c3c")
        self.lbl_kpi_yield = self._create_kpi_card("YIELD (PASS RATE)", "100.0%", "#f1c40f")

        right_layout.addWidget(self.lbl_kpi_total["box"])
        right_layout.addWidget(self.lbl_kpi_good["box"])
        right_layout.addWidget(self.lbl_kpi_bad["box"])
        right_layout.addWidget(self.lbl_kpi_yield["box"])

        # Manual Trigger Button for testing
        btn_manual_trigger = QPushButton("⚡  SIMULATE MODBUS TRIGGER")
        btn_manual_trigger.setObjectName("BtnTrigger")
        btn_manual_trigger.setFixedHeight(45)
        btn_manual_trigger.clicked.connect(self._on_manual_trigger_clicked)

        btn_reset_stats = QPushButton("🔄  Reset Production Counters")
        btn_reset_stats.clicked.connect(self._reset_kpi_counters)

        right_layout.addWidget(btn_manual_trigger)
        right_layout.addWidget(btn_reset_stats)
        right_layout.addStretch()

        layout.addWidget(left_box, stretch=6)
        layout.addWidget(right_box, stretch=4)

        return widget

    def _create_kpi_card(self, title_text, initial_val, color_hex):
        box = QFrame()
        box.setObjectName("KPICard")
        b_layout = QVBoxLayout(box)
        b_layout.setContentsMargins(15, 10, 15, 10)

        t_lbl = QLabel(title_text)
        t_lbl.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        t_lbl.setStyleSheet("color: #8899a6;")

        v_lbl = QLabel(initial_val)
        v_lbl.setFont(QFont("Segoe UI", 22, QFont.Weight.Bold))
        v_lbl.setStyleSheet(f"color: {color_hex};")

        b_layout.addWidget(t_lbl)
        b_layout.addWidget(v_lbl)

        return {"box": box, "val_label": v_lbl}

    # ──────────────────────────────────────────
    # TAB 2: MODBUS & CAMERA SETUP
    # ──────────────────────────────────────────
    def _build_tab_modbus(self):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setSpacing(15)

        # Config Column
        cfg_box = QGroupBox("MODBUS TCP PARAMETERS & MODE")
        cfg_layout = QVBoxLayout(cfg_box)

        # Mode Selection: Server vs Client
        f_mode = QHBoxLayout()
        f_mode.addWidget(QLabel("Operating Mode:"))
        self.cmb_modbus_mode = QComboBox()
        self.cmb_modbus_mode.addItems(["Server", "Client"])
        self.cmb_modbus_mode.currentTextChanged.connect(self._on_modbus_mode_changed)
        f_mode.addWidget(self.cmb_modbus_mode)
        cfg_layout.addLayout(f_mode)

        # IP & Port
        f_ip = QHBoxLayout()
        f_ip.addWidget(QLabel("IP Address / Host:"))
        self.txt_modbus_host = QLineEdit("127.0.0.1")
        f_ip.addWidget(self.txt_modbus_host)
        cfg_layout.addLayout(f_ip)

        f_port = QHBoxLayout()
        f_port.addWidget(QLabel("TCP Port:"))
        self.spn_modbus_port = QSpinBox()
        self.spn_modbus_port.setRange(1, 65535)
        self.spn_modbus_port.setValue(5020)
        f_port.addWidget(self.spn_modbus_port)

        f_unit = QHBoxLayout()
        f_unit.addWidget(QLabel("Unit / Slave ID:"))
        self.spn_modbus_unit = QSpinBox()
        self.spn_modbus_unit.setRange(1, 255)
        self.spn_modbus_unit.setValue(1)
        f_unit.addWidget(self.spn_modbus_unit)

        cfg_layout.addLayout(f_port)
        cfg_layout.addLayout(f_unit)

        # Coil Registers
        grp_coils = QGroupBox("COIL REGISTERS MAPPING")
        c_layout = QVBoxLayout(grp_coils)

        f_trig = QHBoxLayout()
        f_trig.addWidget(QLabel("Trigger Coil (Input):"))
        self.spn_coil_trigger = QSpinBox()
        self.spn_coil_trigger.setRange(0, 9999)
        self.spn_coil_trigger.setValue(1)
        f_trig.addWidget(self.spn_coil_trigger)
        c_layout.addLayout(f_trig)

        f_good = QHBoxLayout()
        f_good.addWidget(QLabel("Good Coil (Output, ON=Good):"))
        self.spn_coil_good = QSpinBox()
        self.spn_coil_good.setRange(0, 9999)
        self.spn_coil_good.setValue(10)
        f_good.addWidget(self.spn_coil_good)
        c_layout.addLayout(f_good)

        f_bad = QHBoxLayout()
        f_bad.addWidget(QLabel("Bad Coil (Output, ON=Bad):"))
        self.spn_coil_bad = QSpinBox()
        self.spn_coil_bad.setRange(0, 9999)
        self.spn_coil_bad.setValue(12)
        f_bad.addWidget(self.spn_coil_bad)
        c_layout.addLayout(f_bad)

        f_busy = QHBoxLayout()
        f_busy.addWidget(QLabel("Busy Coil (Output, ON=Busy):"))
        self.spn_coil_busy = QSpinBox()
        self.spn_coil_busy.setRange(0, 9999)
        self.spn_coil_busy.setValue(11)
        f_busy.addWidget(self.spn_coil_busy)
        c_layout.addLayout(f_busy)

        cfg_layout.addWidget(grp_coils)

        # Timing
        f_timing = QHBoxLayout()
        f_timing.addWidget(QLabel("Result Display Hold Time (sec):"))
        self.spn_hold_time = QSpinBox()
        self.spn_hold_time.setRange(1, 60)
        self.spn_hold_time.setValue(5)
        self.spn_hold_time.valueChanged.connect(self._on_hold_time_changed)
        f_timing.addWidget(self.spn_hold_time)
        cfg_layout.addLayout(f_timing)

        f_poll = QHBoxLayout()
        f_poll.addWidget(QLabel("Poll Interval (ms):"))
        self.spn_poll_interval = QSpinBox()
        self.spn_poll_interval.setRange(10, 1000)
        self.spn_poll_interval.setValue(50)
        f_poll.addWidget(self.spn_poll_interval)
        cfg_layout.addLayout(f_poll)

        # Camera Index
        f_cam = QHBoxLayout()
        f_cam.addWidget(QLabel("Camera USB Index:"))
        self.spn_camera_idx = QSpinBox()
        self.spn_camera_idx.setRange(0, 10)
        self.spn_camera_idx.setValue(0)
        f_cam.addWidget(self.spn_camera_idx)
        cfg_layout.addLayout(f_cam)

        btn_apply_modbus = QPushButton("💾  Apply & Restart Modbus Engine")
        btn_apply_modbus.setObjectName("BtnApply")
        btn_apply_modbus.clicked.connect(self._restart_modbus_worker)
        cfg_layout.addWidget(btn_apply_modbus)

        btn_reset_coils = QPushButton("🔄  Reset All Coils to 0 (Force Idle)")
        btn_reset_coils.clicked.connect(self._force_reset_coils)
        cfg_layout.addWidget(btn_reset_coils)
        cfg_layout.addStretch()

        # Right Column: Live Coil State, Watch Window & Log Console
        right_box = QGroupBox("MODBUS MONITOR, WATCH & LOG")
        r_layout = QVBoxLayout(right_box)

        # Connection Info Banner
        self.lbl_conn_detail = QLabel("Connection: Not started")
        self.lbl_conn_detail.setObjectName("ConnBanner")
        self.lbl_conn_detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_conn_detail.setFixedHeight(30)
        r_layout.addWidget(self.lbl_conn_detail)

        # Coil State Indicator Cards
        indicators_frame = QFrame()
        ind_layout = QHBoxLayout(indicators_frame)
        self.lbl_ind_trig = QLabel("TRIGGER COIL\n[ OFF ]")
        self.lbl_ind_trig.setObjectName("CoilOff")
        self.lbl_ind_trig.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.lbl_ind_good = QLabel("GOOD COIL\n[ OFF ]")
        self.lbl_ind_good.setObjectName("CoilOff")
        self.lbl_ind_good.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.lbl_ind_bad = QLabel("BAD COIL\n[ OFF ]")
        self.lbl_ind_bad.setObjectName("CoilOff")
        self.lbl_ind_bad.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.lbl_ind_busy = QLabel("BUSY COIL\n[ OFF ]")
        self.lbl_ind_busy.setObjectName("CoilOff")
        self.lbl_ind_busy.setAlignment(Qt.AlignmentFlag.AlignCenter)

        ind_layout.addWidget(self.lbl_ind_trig)
        ind_layout.addWidget(self.lbl_ind_good)
        ind_layout.addWidget(self.lbl_ind_bad)
        ind_layout.addWidget(self.lbl_ind_busy)
        r_layout.addWidget(indicators_frame)

        # Live Register Watch Table
        watch_box = QGroupBox("LIVE COIL REGISTER WATCH (Real-time)")
        w_layout = QVBoxLayout(watch_box)
        self.tbl_watch = QTableWidget(0, 3)
        self.tbl_watch.setHorizontalHeaderLabels(["Coil Address", "Value", "Mapped To"])
        self.tbl_watch.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tbl_watch.setMaximumHeight(200)
        w_layout.addWidget(self.tbl_watch)
        r_layout.addWidget(watch_box)

        # Log Console
        self.txt_modbus_log = QTextEdit()
        self.txt_modbus_log.setReadOnly(True)
        r_layout.addWidget(self.txt_modbus_log)

        layout.addWidget(cfg_box, stretch=4)
        layout.addWidget(right_box, stretch=6)

        return widget

    # ──────────────────────────────────────────
    # TAB 3: MODEL & TRAINING
    # ──────────────────────────────────────────
    def _build_tab_model(self):
        widget = QWidget()
        layout = QHBoxLayout(widget)

        # Model Info & Trainer Setup
        cfg_box = QGroupBox("MODEL & TRAINING CONTROL")
        cfg_layout = QVBoxLayout(cfg_box)

        # Active Model File
        f_mod = QHBoxLayout()
        f_mod.addWidget(QLabel("Model File Path:"))
        self.txt_model_path = QLineEdit(DEFAULT_MODEL_PATH)
        btn_browse_mod = QPushButton("Browse...")
        btn_browse_mod.clicked.connect(self._browse_model_file)
        f_mod.addWidget(self.txt_model_path)
        f_mod.addWidget(btn_browse_mod)
        cfg_layout.addLayout(f_mod)

        btn_load_mod = QPushButton("Reload Model File")
        btn_load_mod.clicked.connect(self._reload_model_file)
        cfg_layout.addWidget(btn_load_mod)

        # Dataset Folder
        f_data = QHBoxLayout()
        f_data.addWidget(QLabel("Dataset Directory:"))
        self.txt_data_dir = QLineEdit(DEFAULT_DATA_DIR)
        btn_browse_data = QPushButton("Browse...")
        btn_browse_data.clicked.connect(self._browse_data_dir)
        f_data.addWidget(self.txt_data_dir)
        f_data.addWidget(btn_browse_data)
        cfg_layout.addLayout(f_data)

        # Hyperparameters
        f_ep = QHBoxLayout()
        f_ep.addWidget(QLabel("Training Epochs:"))
        self.spn_epochs = QSpinBox()
        self.spn_epochs.setRange(1, 200)
        self.spn_epochs.setValue(25)
        f_ep.addWidget(self.spn_epochs)

        f_batch = QHBoxLayout()
        f_batch.addWidget(QLabel("Batch Size:"))
        self.spn_batch_size = QSpinBox()
        self.spn_batch_size.setRange(1, 128)
        self.spn_batch_size.setValue(16)
        f_batch.addWidget(self.spn_batch_size)

        cfg_layout.addLayout(f_ep)
        cfg_layout.addLayout(f_batch)

        self.btn_start_train = QPushButton("🚀  Start Model Training")
        self.btn_start_train.setObjectName("BtnApply")
        self.btn_start_train.clicked.connect(self._start_training)
        cfg_layout.addWidget(self.btn_start_train)
        cfg_layout.addStretch()

        # Log & Progress
        log_box = QGroupBox("TRAINING PROGRESS & LOG CONSOLE")
        log_layout = QVBoxLayout(log_box)

        self.txt_train_log = QTextEdit()
        self.txt_train_log.setReadOnly(True)

        log_layout.addWidget(self.txt_train_log)

        layout.addWidget(cfg_box, stretch=4)
        layout.addWidget(log_box, stretch=6)

        return widget

    # ──────────────────────────────────────────
    # TAB 4: PRODUCTION HISTORY
    # ──────────────────────────────────────────
    def _build_tab_history(self):
        widget = QWidget()
        layout = QHBoxLayout(widget)

        # Table
        table_box = QGroupBox("INSPECTION HISTORY LOG")
        t_layout = QVBoxLayout(table_box)

        self.tbl_history = QTableWidget(0, 5)
        self.tbl_history.setHorizontalHeaderLabels(["Timestamp", "Result", "Confidence", "Image Saved", "Trigger Source"])
        self.tbl_history.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tbl_history.itemSelectionChanged.connect(self._on_history_row_selected)
        t_layout.addWidget(self.tbl_history)

        # Preview side
        prev_box = QGroupBox("HISTORICAL IMAGE PREVIEW")
        p_layout = QVBoxLayout(prev_box)

        self.lbl_history_image = QLabel("Select a row from the log table to preview image.")
        self.lbl_history_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_history_image.setObjectName("ImageDisplayCard")
        p_layout.addWidget(self.lbl_history_image)

        layout.addWidget(table_box, stretch=6)
        layout.addWidget(prev_box, stretch=4)

        return widget

    # ──────────────────────────────────────────
    # HMI Logic & 5-Second Auto-Reset
    # ──────────────────────────────────────────
    def _set_idle_image_display(self):
        canvas = QPixmap(640, 480)
        canvas.fill(QColor("#1e272e"))
        self.lbl_image_display.setPixmap(canvas)

    def _on_manual_trigger_clicked(self):
        if self.modbus_worker and self.modbus_worker.isRunning():
            cfg = self.modbus_worker.config
            camera_idx = int(cfg.get("camera_idx", 0))

            # Simulate trigger processing
            threading.Thread(
                target=self.modbus_worker._process_inspection,
                kwargs={
                    "camera_idx": camera_idx,
                    "write_good_fn": lambda v: None,
                    "write_bad_fn": lambda v: None,
                    "write_busy_fn": lambda v: None,
                    "reset_trig_fn": lambda: None,
                    "trigger_source": "Manual GUI Test Button"
                },
                daemon=True
            ).start()

    def _handle_trigger_detected(self, timestamp_str):
        self.reset_timer.stop()
        self.lbl_countdown.setText("")
        self.lbl_result_banner.setText("⚡ TRIGGER DETECTED — CLASSIFYING...")
        self.lbl_result_banner.setObjectName("BannerProcessing")
        self.lbl_result_banner.setStyle(self.lbl_result_banner.style())
        self._handle_status_changed("PROCESSING - Capturing Image & Classifying...", "processing")

    def _handle_inspection_completed(self, data):
        label = data["label"].upper()
        conf = data["confidence"]
        is_good = data["is_good"]
        bgr_frame = data["bgr_frame"]

        # 1. Update KPI Counters
        self.total_count += 1
        if is_good:
            self.good_count += 1
        else:
            self.bad_count += 1

        yield_pct = (self.good_count / self.total_count) * 100.0 if self.total_count > 0 else 100.0

        self.lbl_kpi_total["val_label"].setText(str(self.total_count))
        self.lbl_kpi_good["val_label"].setText(str(self.good_count))
        self.lbl_kpi_bad["val_label"].setText(str(self.bad_count))
        self.lbl_kpi_yield["val_label"].setText(f"{yield_pct:.1f}%")

        # 2. Display Image on HMI Screen
        h, w, ch = bgr_frame.shape
        bytes_per_line = ch * w
        q_img = QImage(bgr_frame.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
        pix = QPixmap.fromImage(q_img).scaled(
            self.lbl_image_display.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.lbl_image_display.setPixmap(pix)

        # 3. Display Result Status Banner
        if is_good:
            self.lbl_result_banner.setText(f"✅  RESULT: GOOD ({conf:.1f}% Confidence)")
            self.lbl_result_banner.setObjectName("BannerGood")
        else:
            self.lbl_result_banner.setText(f"❌  RESULT: BAD ({conf:.1f}% Confidence)")
            self.lbl_result_banner.setObjectName("BannerBad")
        self.lbl_result_banner.setStyle(self.lbl_result_banner.style())

        # 4. Add to History Table
        row = self.tbl_history.rowCount()
        self.tbl_history.insertRow(row)
        self.tbl_history.setItem(row, 0, QTableWidgetItem(data["timestamp"]))
        self.tbl_history.setItem(row, 1, QTableWidgetItem(label))
        self.tbl_history.setItem(row, 2, QTableWidgetItem(f"{conf:.1f}%"))
        self.tbl_history.setItem(row, 3, QTableWidgetItem(data["image_path"]))
        self.tbl_history.setItem(row, 4, QTableWidgetItem(data["source"]))

        # 5. Start 5-second Countdown to Return to Idle
        self.countdown_seconds = self.hold_time_setting
        self.reset_timer.start()
        self._handle_status_changed(f"RESULT READY ({label}) — HOLDING {self.hold_time_setting:.0f}s", "listening")

    def _on_reset_timer_tick(self):
        self.countdown_seconds -= 0.1
        if self.countdown_seconds <= 0:
            self.reset_timer.stop()
            self.lbl_countdown.setText("")
            self.lbl_result_banner.setText("IDLE — WAITING FOR MODBUS TRIGGER")
            self.lbl_result_banner.setObjectName("BannerIdle")
            self.lbl_result_banner.setStyle(self.lbl_result_banner.style())
            self._set_idle_image_display()

            mode_str = self.cmb_modbus_mode.currentText()
            if mode_str == "Server":
                self._handle_status_changed("IDLE — SERVER LISTENING", "listening")
            else:
                self._handle_status_changed("IDLE — CLIENT CONNECTED", "connected")

            # Zero out all Modbus coil registers and re-arm trigger for next rising edge
            if self.modbus_worker and self.modbus_worker.isRunning():
                self.modbus_worker.request_reset_all_coils()
        else:
            self.lbl_countdown.setText(f"⏱ Returning to Idle Mode in {self.countdown_seconds:.1f}s...")

    def _force_reset_coils(self):
        """Manual button action to force all coils to 0."""
        if self.modbus_worker and self.modbus_worker.isRunning():
            self.modbus_worker.request_reset_all_coils()
            self._log_modbus("[+] Force reset requested — All coils set to 0.")

    def _reset_kpi_counters(self):
        self.total_count = 0
        self.good_count = 0
        self.bad_count = 0
        self.lbl_kpi_total["val_label"].setText("0")
        self.lbl_kpi_good["val_label"].setText("0")
        self.lbl_kpi_bad["val_label"].setText("0")
        self.lbl_kpi_yield["val_label"].setText("100.0%")

    # ──────────────────────────────────────────
    # Modbus Worker Control
    # ──────────────────────────────────────────
    def _restart_modbus_worker(self):
        if self.modbus_worker and self.modbus_worker.isRunning():
            self.modbus_worker.stop()
            self.modbus_worker.wait(1000)

        cfg = {
            "mode": self.cmb_modbus_mode.currentText(),
            "host": self.txt_modbus_host.text().strip(),
            "port": self.spn_modbus_port.value(),
            "unit_id": self.spn_modbus_unit.value(),
            "trigger_coil": self.spn_coil_trigger.value(),
            "good_coil": self.spn_coil_good.value(),
            "bad_coil": self.spn_coil_bad.value(),
            "busy_coil": self.spn_coil_busy.value(),
            "poll_interval_ms": self.spn_poll_interval.value(),
            "camera_idx": self.spn_camera_idx.value(),
            "production_dir": DEFAULT_PRODUCTION_DIR,
        }

        self.lbl_mode_badge.setText(f"MODE: {cfg['mode'].upper()}")

        self.modbus_worker = ModbusWorker(cfg, self.classifier)
        self.modbus_worker.trigger_detected.connect(self._handle_trigger_detected)
        self.modbus_worker.inspection_completed.connect(self._handle_inspection_completed)
        self.modbus_worker.status_changed.connect(self._handle_status_changed)
        self.modbus_worker.modbus_log_signal.connect(self._log_modbus)
        self.modbus_worker.coil_state_signal.connect(self._update_coil_indicators)
        self.modbus_worker.watch_data_signal.connect(self._update_watch_table)
        self.modbus_worker.connection_info_signal.connect(self._update_connection_info)
        self.modbus_worker.start()

    def _handle_status_changed(self, text, style_class):
        self.lbl_conn_status.setText(f"STATUS: {text}")
        if style_class in ("listening", "connected", "online"):
            self.lbl_conn_status.setObjectName("BadgeStatusOnline")
        elif style_class == "processing":
            self.lbl_conn_status.setObjectName("BadgeStatusBusy")
        elif style_class == "idle":
            self.lbl_conn_status.setObjectName("BadgeStatusIdle")
        else:
            self.lbl_conn_status.setObjectName("BadgeStatusError")
        self.lbl_conn_status.style().unpolish(self.lbl_conn_status)
        self.lbl_conn_status.style().polish(self.lbl_conn_status)

    def _log_modbus(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self.txt_modbus_log.append(f"[{ts}] {msg}")

    def _update_coil_indicators(self, trig, good, bad, busy):
        self.lbl_ind_trig.setText(f"TRIGGER COIL ({self.spn_coil_trigger.value()})\n[ {'ON' if trig else 'OFF'} ]")
        self.lbl_ind_trig.setObjectName("CoilOn" if trig else "CoilOff")
        self.lbl_ind_trig.setStyle(self.lbl_ind_trig.style())

        self.lbl_ind_good.setText(f"GOOD COIL ({self.spn_coil_good.value()})\n[ {'ON ✔' if good else 'OFF'} ]")
        self.lbl_ind_good.setObjectName("CoilGood" if good else "CoilOff")
        self.lbl_ind_good.setStyle(self.lbl_ind_good.style())

        self.lbl_ind_bad.setText(f"BAD COIL ({self.spn_coil_bad.value()})\n[ {'ON ✘' if bad else 'OFF'} ]")
        self.lbl_ind_bad.setObjectName("CoilBad" if bad else "CoilOff")
        self.lbl_ind_bad.setStyle(self.lbl_ind_bad.style())

        self.lbl_ind_busy.setText(f"BUSY COIL ({self.spn_coil_busy.value()})\n[ {'ON' if busy else 'OFF'} ]")
        self.lbl_ind_busy.setObjectName("CoilBusy" if busy else "CoilOff")
        self.lbl_ind_busy.setStyle(self.lbl_ind_busy.style())

    def _update_watch_table(self, watch_data: dict):
        """Update the live register watch table with current coil values."""
        trigger_coil = self.spn_coil_trigger.value()
        good_coil = self.spn_coil_good.value()
        bad_coil = self.spn_coil_bad.value()
        busy_coil = self.spn_coil_busy.value()

        # Build mapping labels
        label_map = {
            trigger_coil: "TRIGGER (Input)",
            good_coil: "GOOD (Output)",
            bad_coil: "BAD (Output)",
            busy_coil: "BUSY (Output)",
        }

        self.tbl_watch.setRowCount(len(watch_data))
        for row, (addr, val) in enumerate(sorted(watch_data.items())):
            self.tbl_watch.setItem(row, 0, QTableWidgetItem(str(addr)))

            val_item = QTableWidgetItem("ON (1)" if val else "OFF (0)")
            if val:
                val_item.setBackground(QColor("#27ae60"))
                val_item.setForeground(QColor("#ffffff"))
            else:
                val_item.setBackground(QColor("#2c3e50"))
                val_item.setForeground(QColor("#95a5a6"))
            self.tbl_watch.setItem(row, 1, val_item)

            mapped = label_map.get(addr, "")
            map_item = QTableWidgetItem(mapped)
            if mapped:
                map_item.setForeground(QColor("#00d2d3"))
            self.tbl_watch.setItem(row, 2, map_item)

    def _update_connection_info(self, info: dict):
        """Update the connection info banner."""
        mode = info.get("mode", "?")
        host = info.get("host", "?")
        port = info.get("port", "?")
        status = info.get("status", "?")
        unit_id = info.get("unit_id", "-")

        text = f"{mode}  |  {host}:{port}  |  Unit ID: {unit_id}  |  Status: {status}"
        self.lbl_conn_detail.setText(text)

        if status in ("Listening", "Connected"):
            self.lbl_conn_detail.setStyleSheet(
                "background-color: #27ae60; color: #ffffff; font-weight: bold; "
                "border-radius: 4px; padding: 4px;"
            )
        elif status in ("DISCONNECTED", "ERROR"):
            self.lbl_conn_detail.setStyleSheet(
                "background-color: #c0392b; color: #ffffff; font-weight: bold; "
                "border-radius: 4px; padding: 4px;"
            )
        else:
            self.lbl_conn_detail.setStyleSheet(
                "background-color: #2c3e50; color: #bdc3c7; font-weight: bold; "
                "border-radius: 4px; padding: 4px;"
            )

    def _on_modbus_mode_changed(self, mode_str):
        if mode_str == "Server":
            self.txt_modbus_host.setText("127.0.0.1")
        else:
            self.txt_modbus_host.setText("127.0.0.1")

    def _on_hold_time_changed(self, val):
        self.hold_time_setting = float(val)

    # ──────────────────────────────────────────
    # Model & Training Actions
    # ──────────────────────────────────────────
    def _browse_model_file(self):
        fpath, _ = QFileDialog.getOpenFileName(self, "Select Model File", SCRIPT_DIR, "PyTorch Model (*.pth)")
        if fpath:
            self.txt_model_path.setText(fpath)

    def _browse_data_dir(self):
        dpath = QFileDialog.getExistingDirectory(self, "Select Dataset Folder", SCRIPT_DIR)
        if dpath:
            self.txt_data_dir.setText(dpath)

    def _reload_model_file(self):
        fpath = self.txt_model_path.text().strip()
        ok = self.classifier.load_model(fpath)
        if ok:
            QMessageBox.information(self, "Success", f"Model successfully reloaded:\n{fpath}")
        else:
            QMessageBox.warning(self, "Error", f"Failed to load model file:\n{fpath}")

    def _start_training(self):
        data_dir = self.txt_data_dir.text().strip()
        model_path = self.txt_model_path.text().strip()
        epochs = self.spn_epochs.value()
        batch_size = self.spn_batch_size.value()

        self.btn_start_train.setEnabled(False)
        self.txt_train_log.append("=== STARTING TRAINING PROCESS ===")

        self.train_thread = TrainingThread(data_dir, model_path, epochs, batch_size, 0.001)
        self.train_thread.progress_signal.connect(self._log_training)
        self.train_thread.finished_signal.connect(self._on_training_finished)
        self.train_thread.start()

    def _log_training(self, text):
        self.txt_train_log.append(text)

    def _on_training_finished(self, success, msg):
        self.btn_start_train.setEnabled(True)
        self.txt_train_log.append(f"\n{msg}\n")
        if success:
            self.classifier.load_model(self.txt_model_path.text().strip())
            QMessageBox.information(self, "Training Complete", msg)
        else:
            QMessageBox.critical(self, "Training Error", msg)

    # ──────────────────────────────────────────
    # History Table Selection
    # ──────────────────────────────────────────
    def _on_history_row_selected(self):
        rows = self.tbl_history.selectedItems()
        if not rows:
            return
        row = rows[0].row()
        img_path = self.tbl_history.item(row, 3).text()

        if os.path.isfile(img_path):
            pix = QPixmap(img_path).scaled(
                self.lbl_history_image.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.lbl_history_image.setPixmap(pix)
        else:
            self.lbl_history_image.setText(f"File not found: {img_path}")

    # ──────────────────────────────────────────
    # Modern Industrial Dark Theme Styling (QSS)
    # ──────────────────────────────────────────
    def _apply_dark_theme(self):
        qss = """
        QMainWindow {
            background-color: #0f141d;
            color: #d1d8e0;
            font-family: "Segoe UI", sans-serif;
        }

        #HeaderFrame {
            background-color: #1a222d;
            border-bottom: 2px solid #2c3e50;
            border-radius: 6px;
        }

        QLabel {
            color: #d1d8e0;
        }

        QGroupBox {
            background-color: #171f2a;
            border: 1px solid #283646;
            border-radius: 8px;
            margin-top: 12px;
            font-size: 11px;
            font-weight: bold;
            color: #4bc235;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 2px 8px;
        }

        QTabWidget::pane {
            border: 1px solid #283646;
            background-color: #121922;
            border-radius: 8px;
        }
        QTabBar::tab {
            background-color: #171f2a;
            color: #8899a6;
            padding: 10px 20px;
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
            margin-right: 2px;
            font-weight: bold;
        }
        QTabBar::tab:selected {
            background-color: #243346;
            color: #00d2d3;
            border-bottom: 3px solid #00d2d3;
        }

        #ImageDisplayCard {
            background-color: #090c10;
            border: 2px solid #283646;
            border-radius: 10px;
        }

        #KPICard {
            background-color: #1a2432;
            border: 1px solid #2c3d52;
            border-radius: 8px;
        }

        #BannerIdle {
            background-color: #1e2936;
            color: #8899a6;
            border: 2px solid #34495e;
            border-radius: 8px;
        }
        #BannerProcessing {
            background-color: #f39c12;
            color: #ffffff;
            border: 2px solid #e67e22;
            border-radius: 8px;
        }
        #BannerGood {
            background-color: #27ae60;
            color: #ffffff;
            border: 2px solid #2ecc71;
            border-radius: 8px;
        }
        #BannerBad {
            background-color: #c0392b;
            color: #ffffff;
            border: 2px solid #e74c3c;
            border-radius: 8px;
        }

        #BadgeMode {
            background-color: #2980b9;
            color: #ffffff;
            font-weight: bold;
            padding: 4px 10px;
            border-radius: 12px;
        }
        #BadgeStatusOnline {
            background-color: #27ae60;
            color: #ffffff;
            font-weight: bold;
            padding: 4px 10px;
            border-radius: 12px;
        }
        #BadgeStatusIdle {
            background-color: #34495e;
            color: #ffffff;
            font-weight: bold;
            padding: 4px 10px;
            border-radius: 12px;
        }
        #BadgeStatusBusy {
            background-color: #f39c12;
            color: #ffffff;
            font-weight: bold;
            padding: 4px 10px;
            border-radius: 12px;
        }
        #BadgeStatusError {
            background-color: #c0392b;
            color: #ffffff;
            font-weight: bold;
            padding: 4px 10px;
            border-radius: 12px;
        }

        #CoilOff {
            background-color: #2c3e50;
            color: #bdc3c7;
            padding: 10px;
            border-radius: 6px;
            font-weight: bold;
        }
        #CoilOn {
            background-color: #f39c12;
            color: #ffffff;
            padding: 10px;
            border-radius: 6px;
            font-weight: bold;
        }
        #CoilGood {
            background-color: #27ae60;
            color: #ffffff;
            padding: 10px;
            border-radius: 6px;
            font-weight: bold;
        }
        #CoilBad {
            background-color: #c0392b;
            color: #ffffff;
            padding: 10px;
            border-radius: 6px;
            font-weight: bold;
        }
        #CoilBusy {
            background-color: #e67e22;
            color: #ffffff;
            padding: 10px;
            border-radius: 6px;
            font-weight: bold;
        }

        QPushButton {
            background-color: #243346;
            color: #ffffff;
            border: 1px solid #34495e;
            border-radius: 6px;
            padding: 8px 16px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #34495e;
        }
        #BtnTrigger {
            background-color: #8e44ad;
            border: 1px solid #9b59b6;
            font-size: 13px;
        }
        #BtnTrigger:hover {
            background-color: #9b59b6;
        }
        #BtnApply {
            background-color: #27ae60;
            border: 1px solid #2ecc71;
            font-size: 13px;
        }
        #BtnApply:hover {
            background-color: #2ecc71;
        }

        QLineEdit, QSpinBox, QComboBox, QTextEdit {
            background-color: #0f141d;
            color: #00d2d3;
            border: 1px solid #283646;
            border-radius: 4px;
            padding: 6px;
        }
        QTableWidget {
            background-color: #0f141d;
            color: #d1d8e0;
            gridline-color: #283646;
            border-radius: 6px;
        }
        QHeaderView::section {
            background-color: #171f2a;
            color: #00d2d3;
            padding: 6px;
            border: 1px solid #283646;
            font-weight: bold;
        }
        """
        self.setStyleSheet(qss)

    def closeEvent(self, event):
        if self.modbus_worker:
            self.modbus_worker.stop()
            self.modbus_worker.wait(1000)
        event.accept()


# ──────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
