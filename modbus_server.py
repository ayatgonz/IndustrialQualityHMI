"""
Modbus TCP/IP Classifier Server (Object ROI, Colors, Text & Shades)
=====================================================================
Runs a Modbus TCP server that:
  1. Monitors a trigger coil for a rising edge (0 → 1)
  2. On trigger: captures a webcam photo, isolates the object from background,
     extracts spatial, color, and text metrics, and classifies it (Good/Bad)
  3. Writes the result to a result coil (1 = Good, 0 = Bad)
  4. Saves every captured image to a "production/" folder
  5. Resets the trigger coil and keeps listening

Usage:
    python modbus_server.py
    python modbus_server.py --host 0.0.0.0 --port 502 --trigger-coil 1 --result-coil 10
    python modbus_server.py --camera 1 --model my_model.pth

Modbus Map:
    Coil 1  (default)  →  Trigger   (PLC writes 1 to request classification)
    Coil 10 (default)  →  Result    (Server writes 1=Good, 0=Bad)
    Coil 11            →  Busy flag (Server writes 1 while classifying)
"""

import argparse
import os
import sys
import time
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from pymodbus.server import StartTcpServer
from pymodbus.datastore import (
    ModbusSlaveContext,
    ModbusServerContext,
    ModbusSequentialDataBlock,
)
from pymodbus.device import ModbusDeviceIdentification


# ──────────────────────────────────────────────
# Config (must match train.py)
# ──────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(SCRIPT_DIR, "model.pth")
DEFAULT_PRODUCTION_DIR = os.path.join(SCRIPT_DIR, "production")
INPUT_SIZE = 256


# ──────────────────────────────────────────────
# Object vs Background Separation (ROI Extractor)
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


# ──────────────────────────────────────────────
# Explicit Color, Shade & Text Sharpness Extractor
# ──────────────────────────────────────────────
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


# ──────────────────────────────────────────────
# Model Definition
# ──────────────────────────────────────────────
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


def build_model(num_classes: int = 2):
    return HybridClassifier(num_classes=num_classes)


preprocess = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_classifier(model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    label_map = checkpoint.get("label_map", {"good": 0, "bad": 1})
    idx_to_label = {v: k for k, v in label_map.items()}

    model = build_model(num_classes=len(label_map))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    val_acc = checkpoint.get("val_acc", "N/A")
    print(f"[+] Model loaded: {model_path} (val acc: {val_acc}%)")
    return model, idx_to_label, device


def classify_image(image: Image.Image, model, idx_to_label, device):
    roi_image = extract_object_roi(image)
    color_stats = get_color_and_text_stats(roi_image).unsqueeze(0).to(device)
    spatial_tensor = preprocess(roi_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(spatial_tensor, color_stats)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred_idx = probs.max(1)

    label = idx_to_label[pred_idx.item()]
    return label, confidence.item() * 100


# ──────────────────────────────────────────────
# Webcam capture
# ──────────────────────────────────────────────
def capture_from_webcam(camera_index: int = 0, warmup_frames: int = 15):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[-] Cannot open camera {camera_index}")
        return None

    for _ in range(warmup_frames):
        cap.read()
        time.sleep(0.02)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("[-] Failed to read frame")
        return None

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb), frame


# ──────────────────────────────────────────────
# Rising-edge monitor thread
# ──────────────────────────────────────────────
class TriggerMonitor(threading.Thread):
    def __init__(self, context, trigger_coil, result_coil, busy_coil,
                 camera_index, model, idx_to_label, device,
                 production_dir, poll_interval=0.05):
        super().__init__(daemon=True)
        self.context = context
        self.trigger_coil = trigger_coil
        self.result_coil = result_coil
        self.busy_coil = busy_coil
        self.camera_index = camera_index
        self.model = model
        self.idx_to_label = idx_to_label
        self.device = device
        self.production_dir = Path(production_dir)
        self.production_dir.mkdir(parents=True, exist_ok=True)
        self.poll_interval = poll_interval
        self.prev_trigger = False
        self.running = True
        self.capture_count = 0

    def _read_coil(self, address):
        slave = self.context[0x00]
        values = slave.getValues(1, address, count=1)
        return bool(values[0])

    def _write_coil(self, address, value):
        slave = self.context[0x00]
        slave.setValues(1, address, [int(value)])

    def run(self):
        print(f"[+] Monitor started - polling coil {self.trigger_coil} every {self.poll_interval*1000:.0f}ms")
        print(f"    Result -> coil {self.result_coil} | Busy -> coil {self.busy_coil}")
        print(f"    Images saved to: {self.production_dir.resolve()}")
        print("-" * 55)

        while self.running:
            try:
                current = self._read_coil(self.trigger_coil)

                if current and not self.prev_trigger:
                    self._on_trigger()

                self.prev_trigger = current
                time.sleep(self.poll_interval)

            except Exception as e:
                print(f"[!] Monitor error: {e}")
                time.sleep(1)

    def _on_trigger(self):
        self.capture_count += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"\n[+] Trigger #{self.capture_count} detected! [{datetime.now().strftime('%H:%M:%S')}]")

        self._write_coil(self.busy_coil, True)

        try:
            print("    Capturing image...")
            result = capture_from_webcam(self.camera_index)
            if result is None:
                print("    [-] Capture failed - writing Bad (0) to result coil")
                self._write_coil(self.result_coil, False)
                return

            pil_image, bgr_frame = result

            print("    Classifying...")
            label, confidence = classify_image(
                pil_image, self.model, self.idx_to_label, self.device
            )

            is_good = (label == "good")
            status_str = "[GOOD]" if is_good else "[BAD]"

            self._write_coil(self.result_coil, is_good)

            print(f"    Result: {status_str} ({confidence:.1f}% confidence)")
            print(f"    Coil {self.result_coil} <- {'1 (Good)' if is_good else '0 (Bad)'}")

            filename = f"{timestamp}_{label}_{confidence:.0f}pct.jpg"
            save_path = self.production_dir / filename
            cv2.imwrite(str(save_path), bgr_frame)
            print(f"    Image saved: {save_path}")

        except Exception as e:
            print(f"    [-] Error during classification: {e}")
            self._write_coil(self.result_coil, False)

        finally:
            self._write_coil(self.trigger_coil, False)
            self.prev_trigger = False
            self._write_coil(self.busy_coil, False)
            print("    Ready for next trigger")


# ──────────────────────────────────────────────
# Modbus Server Setup
# ──────────────────────────────────────────────
def create_datastore(num_coils=100):
    store = ModbusSlaveContext(
        co=ModbusSequentialDataBlock(0, [0] * (num_coils + 1)),
        di=ModbusSequentialDataBlock(0, [0] * (num_coils + 1)),
        hr=ModbusSequentialDataBlock(0, [0] * (num_coils + 1)),
        ir=ModbusSequentialDataBlock(0, [0] * (num_coils + 1)),
    )
    context = ModbusServerContext(slaves=store, single=True)
    return context


def main():
    parser = argparse.ArgumentParser(
        description="Modbus TCP server that classifies webcam images on trigger.",
    )
    parser.add_argument("--host", type=str, default="127.0.0.5",
                        help="Modbus server bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5020,
                        help="Modbus server TCP port (default: 502)")
    parser.add_argument("--trigger-coil", type=int, default=1,
                        help="Coil address for trigger input (default: 1)")
    parser.add_argument("--result-coil", type=int, default=10,
                        help="Coil address for result output (default: 10)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index (default: 0 = built-in)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH,
                        help="Path to trained model (default: model.pth)")
    parser.add_argument("--production-dir", type=str, default=DEFAULT_PRODUCTION_DIR,
                        help="Folder to save captured images (default: production/)")
    parser.add_argument("--poll-interval", type=float, default=0.05,
                        help="Trigger poll interval in seconds (default: 0.05)")

    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print(f"[-] Model file not found: {args.model}")
        sys.exit(1)

    busy_coil = args.trigger_coil + 1
    if busy_coil == args.result_coil:
        busy_coil = args.result_coil + 1

    model, idx_to_label, device = load_classifier(args.model)

    max_coil = max(args.trigger_coil, args.result_coil, busy_coil) + 10
    context = create_datastore(num_coils=max_coil)

    monitor = TriggerMonitor(
        context=context,
        trigger_coil=args.trigger_coil,
        result_coil=args.result_coil,
        busy_coil=busy_coil,
        camera_index=args.camera,
        model=model,
        idx_to_label=idx_to_label,
        device=device,
        production_dir=args.production_dir,
        poll_interval=args.poll_interval,
    )
    monitor.start()

    identity = ModbusDeviceIdentification()
    identity.VendorName = "ImageClassifier"
    identity.ProductCode = "GoodBadClassifier"
    identity.ProductName = "Webcam Quality Classifier"
    identity.ModelName = "Modbus TCP Classifier Server"

    print()
    print("+-----------------------------------------------+ ")
    print("|       MODBUS TCP CLASSIFIER SERVER            | ")
    print("+-----------------------------------------------+ ")
    print(f"|  Address:       {args.host}:{args.port:<21} |")
    print(f"|  Trigger coil:  {args.trigger_coil:<30} |")
    print(f"|  Result coil:   {args.result_coil:<30} |")
    print(f"|  Busy coil:     {busy_coil:<30} |")
    print(f"|  Camera:        {args.camera:<30} |")
    print(f"|  Model:         {args.model:<30} |")
    print(f"|  Production:    {args.production_dir + '/':<30} |")
    print("+-----------------------------------------------+ ")
    print("|  Write 1 to trigger coil to classify!        |")
    print("|  Press Ctrl+C to stop                        |")
    print("+-----------------------------------------------+ ")
    print()

    try:
        StartTcpServer(
            context=context,
            identity=identity,
            address=(args.host, args.port),
        )
    except PermissionError:
        print(f"\n[-] Permission denied for port {args.port}.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n[-] Server stopped by user.")
        monitor.running = False


if __name__ == "__main__":
    main()
