# Industrial Quality Control — AI Vision & Modbus TCP Inspection HMI

An industrial-grade automated visual quality inspection system featuring a **PyQt6 HMI Dashboard**, a **Hybrid ResNet18 + Color/Text/Shade AI Classifier**, and complete **Modbus TCP Integration** (Server & Client modes) engineered for seamless integration with PLCs, PACs, and industrial automation networks.


---

## Key Industrial Modbus TCP Features

The application is specifically architected for factory automation and PLC line control:

### 1. Dual Operating Modes (Server & Client)
* **Server Mode (PLC Slave)**: Embedded Modbus TCP server listening on a configurable IP/Port (default: `127.0.0.1:5020`). Allows external PLCs (Siemens S7, Allen-Bradley Logix, Schneider Modicon, Beckhoff, etc.) to trigger inspections and read results over Ethernet.
* **Client Mode (PLC Master/Poller)**: Connects to a remote Modbus TCP server/PLC on the network, periodically polling the trigger coil and updating inspection output coils on the PLC.

### 2. Explicit Dual-Coil Result Mapping
Separates inspection results into independent output coils for unambiguous PLC ladder logic integration:
* **Trigger Coil** *(Default Addr 1 - Input)*: Monitored for a rising-edge signal (`0` → `1`) to initiate an image capture and AI classification sequence.
* **Good Coil** *(Default Addr 10 - Output)*: Driven `HIGH` (`1`) when the item passes quality criteria.
* **Bad Coil** *(Default Addr 12 - Output)*: Driven `HIGH` (`1`) when defects, wrong colors, text anomalies, or quality failures are detected.
* **Busy Coil** *(Default Addr 11 - Output)*: Driven `HIGH` (`1`) during camera capture and neural network inference.

### 3. Automatic 5-Second Hold & Idle Register Zeroing
* **Result Display Window**: Holds inspection visual results and output coils high for a configurable duration (default: 5.0 seconds) to allow downstream PLC rejection mechanisms or actuators to read the status.
* **Auto-Zero Register Reset**: Upon timer expiration, all Modbus coils (**Trigger**, **Good**, **Bad**, **Busy**) are automatically zeroed out (`0` / `OFF`), returning the system to **IDLE** mode.
* **Re-armed Rising-Edge Trigger**: The internal edge detector is automatically reset so the system is immediately ready for the next incoming part trigger.

### 4. Real-time Modbus Watch Window & Diagnostics
* **Live Register Watch Table**: Real-time `QTableWidget` polling coil values every ~500ms with color-coded signal states (Green for ON, Slate for OFF) and explicit coil function mapping.
* **Connection Status Banner**: Dynamic header bar badge displaying active operating mode, target IP:Port, Unit/Slave ID, and connection state (`SERVER LISTENING`, `CLIENT CONNECTED`, `PROCESSING`, `IDLE`).
* **Modbus Event Console**: Timestamped log console recording every socket connection, rising edge trigger, classification score, and register write.
* **Manual Force Reset**: One-click `🔄 Reset All Coils to 0 (Force Idle)` button for manual line re-arming and maintenance.

---

## AI Vision Classification Architecture

The core inspection engine uses a **Hybrid Neural Network** designed to eliminate false positives caused by conveyor background noise or color/shade variations:

1. **Object vs. Background ROI Extractor**: Uses Otsu thresholding and contour detection to separate objects from tables, floors, or conveyor background noise before classification.
2. **Explicit Color & Shade Feature Extractor**: Calculates 13 statistical metrics in **LAB** (Lightness, A/B color dimensions) and **HSV** (Hue, Saturation, Value) color spaces alongside **Laplacian edge density** for printed text/character recognition.
3. **ResNet18 Deep Spatial Backbone**: Fuses 512 deep spatial feature maps with the explicit color and sharpness metrics to evaluate both macro visual structures and subtle surface defects.

---

## Application Layout & HMI Tabs

1. **🖥 HMI Dashboard**: Live inspection camera feed, pass/fail result status banner, production KPI counters (**Total**, **Good**, **Bad**, **Yield %**), 5-second countdown timer, and manual trigger test controls.
2. **⚙ Modbus & Camera Setup**: Full network configuration (Server/Client mode, Host IP, Port, Unit ID, Coil address mapping, poll intervals, USB camera index selection, and live Modbus Watch Window).
3. **🧠 Model & Training**: Embedded model trainer with custom dataset folder selector, hyperparameter controls (epochs, batch size), and real-time training log console.
4. **📊 Production History**: Table of historical inspection events with timestamped image previews and log export.

---

## Project Directory Structure

```
Classifier/
├── gui_app.py              # Main PyQt6 Industrial HMI & Modbus GUI application
├── modbus_server.py        # Standalone Modbus TCP server module
├── train.py                # Hybrid ResNet18 + LAB/HSV Color classifier trainer
├── classify_webcam.py      # Standalone webcam classification script
├── create_shortcuts.py     # Windows shortcut generator & icon builder
├── Launch_Classifier.bat   # Launcher script for Windows desktop shortcut
├── app_icon.ico            # High-resolution application icon
├── requirements.txt        # Python package dependencies
├── README.md               # Project documentation
└── dataset/                # Quality dataset directory
    ├── good/               # Training images of acceptable items
    └── bad/                # Training images of defective items
```

---

## Installation & Setup

### Prerequisites
* Windows 10/11
* Python 3.9+ (or virtual environment)

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```
*(Dependencies: `torch`, `torchvision`, `opencv-python`, `Pillow`, `PyQt6`, `pymodbus`, `numpy`)*

### 2. Run the Application
```powershell
python gui_app.py
```

### 3. Create Windows Desktop Launcher Icon
Generate a custom Windows desktop shortcut and Start Menu launcher:
```powershell
python create_shortcuts.py
```

---

## Modbus Coil Configuration Summary

| Coil Function | Default Address | Direction | Description |
|---|---|---|---|
| **Trigger Coil** | `1` | Input | Rising edge (`0` → `1`) triggers inspection |
| **Good Coil** | `10` | Output | Set to `1` when item passes quality criteria |
| **Bad Coil** | `12` | Output | Set to `1` when defect or bad item detected |
| **Busy Coil** | `11` | Output | Set to `1` while camera captures & AI evaluates |

---

## License

Industrial Automation Vision Inspection System — Developed for automated line control and quality assurance.
