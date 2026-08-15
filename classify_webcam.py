"""
Webcam Classifier (Object ROI, Colors, Text & Shades)
=====================================================
Captures a photo from a USB webcam (or built-in camera) and classifies it
as "Good" or "Bad" using the hybrid model from train.py.

Usage:
    python classify_webcam.py                          # built-in cam (index 0)
    python classify_webcam.py --camera 1               # USB webcam at index 1
    python classify_webcam.py --camera 2 --model model.pth
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# ──────────────────────────────────────────────
# Config (must match train.py)
# ──────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(SCRIPT_DIR, "model.pth")
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


# ──────────────────────────────────────────────
# Load Model
# ──────────────────────────────────────────────
def load_model(model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    label_map = checkpoint.get("label_map", {"good": 0, "bad": 1})
    idx_to_label = {v: k for k, v in label_map.items()}

    model = build_model(num_classes=len(label_map))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    val_acc = checkpoint.get("val_acc", "N/A")
    epoch = checkpoint.get("epoch", "N/A")
    print(f"[+] Model loaded from '{model_path}' (epoch {epoch}, val acc {val_acc}%)")

    return model, idx_to_label, device


# ──────────────────────────────────────────────
# Capture photo from webcam
# ──────────────────────────────────────────────
def capture_photo(camera_index: int = 0, warmup_frames: int = 30):
    print(f"[+] Opening camera {camera_index}...")
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"[-] Could not open camera at index {camera_index}.")
        sys.exit(1)

    print("    Warming up camera...")
    for _ in range(warmup_frames):
        cap.read()
        time.sleep(0.03)

    print("   +-----------------------------------------+")
    print("   |  SPACE / ENTER  ->  capture & classify  |")
    print("   |  ESC   / Q      ->  quit                |")
    print("   +-----------------------------------------+")

    captured_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[-] Failed to read frame from camera.")
            break

        display = frame.copy()
        cv2.putText(display, "Press SPACE to capture", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Webcam - Good/Bad Classifier", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (32, 13):
            captured_frame = frame
            print("    [+] Photo captured!")
            break
        elif key in (27, ord('q'), ord('Q')):
            print("    [-] Cancelled by user.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if captured_frame is None:
        return None

    rgb_frame = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_frame)


# ──────────────────────────────────────────────
# Classify
# ──────────────────────────────────────────────
def classify(image: Image.Image, model, idx_to_label, device):
    roi_image = extract_object_roi(image)
    color_stats = get_color_and_text_stats(roi_image).unsqueeze(0).to(device)
    spatial_tensor = preprocess(roi_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(spatial_tensor, color_stats)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = probs.max(1)

    label = idx_to_label[predicted_idx.item()]
    conf = confidence.item() * 100
    return label, conf, probs[0]


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Capture a webcam photo and classify it as Good or Bad.",
    )
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index (default: 0 = built-in webcam)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH,
                        help="Path to the trained model file (default: model.pth)")
    parser.add_argument("--no-preview", action="store_true",
                        help="Skip live preview, capture immediately")
    args = parser.parse_args()

    model, idx_to_label, device = load_model(args.model)

    if args.no_preview:
        print(f"[+] Capturing from camera {args.camera} (no preview)...")
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f"[-] Could not open camera {args.camera}")
            sys.exit(1)
        for _ in range(30):
            cap.read()
            time.sleep(0.03)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("[-] Failed to capture frame.")
            sys.exit(1)
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        print("    [+] Photo captured!")
    else:
        image = capture_photo(camera_index=args.camera)

    if image is None:
        sys.exit(0)

    label, confidence, probs = classify(image, model, idx_to_label, device)

    status_tag = "[GOOD]" if label == "good" else "[BAD]"
    print()
    print("=" * 45)
    print(f"   Result:      {status_tag}")
    print(f"   Confidence:  {confidence:.1f}%")
    print(f"   Probability: (good: {probs[0]:.3f} | bad: {probs[1]:.3f})")
    print("=" * 45)


if __name__ == "__main__":
    main()
