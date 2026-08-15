"""
Hybrid Good vs Bad Classifier (Colors, Textures, Text & Object Segmentation)
=============================================================================
Trains an advanced image classifier that specifically evaluates:
  1. Object vs Background Separation (ROI cropping out table/environment noise)
  2. Colors & Shades (LAB & HSV color space statistical metrics)
  3. Text & Characters / Edge Details (Laplacian edge density + ResNet deep features)

Uses a Hybrid Architecture:
  - Frozen ResNet18 (512 spatial & text features)
  - Explicit Color & Shade Extractor (13 LAB/HSV/Edge features)
  - Fusion Classification Head (525 input dims -> 64 -> 2)

Usage:
    python train.py --epochs 25 --batch_size 16
    python train.py --predict photo.jpg
"""

import argparse
import os
import sys
from pathlib import Path

class DummyStream:
    def write(self, text): pass
    def flush(self): pass

if sys.stdout is None:
    sys.stdout = DummyStream()
if sys.stderr is None:
    sys.stderr = DummyStream()

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler
from torchvision import models, transforms
from PIL import Image


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
if getattr(sys, 'frozen', False):
    SCRIPT_DIR = sys._MEIPASS
else:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(SCRIPT_DIR, "dataset")
DEFAULT_MODEL_PATH = os.path.join(SCRIPT_DIR, "model.pth")

INPUT_SIZE = 256
MODEL_ARCH = "hybrid_resnet18_color_text"


# ──────────────────────────────────────────────
# Object vs Background Separation (ROI Extractor)
# ──────────────────────────────────────────────
def extract_object_roi(pil_img: Image.Image) -> Image.Image:
    """
    Separates the object from the background:
    Detects the main object boundary or crops central ROI to eliminate
    table, floor, or environmental background noise.
    """
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
    """
    Extracts 13 explicit metrics:
      - LAB color space: L (Lightness/Shade), A (Red-Green), B (Yellow-Blue) mean & std
      - HSV color space: H (Hue), S (Saturation), V (Value) mean & std
      - Text / Sharpness: Laplacian edge variance
    """
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
# Dataset
# ──────────────────────────────────────────────
class GoodBadDataset(Dataset):
    """Custom dataset that loads images, applies ROI, and extracts features."""

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    LABEL_MAP = {"good": 0, "bad": 1}

    def __init__(self, data_dir: str, transform=None, log_fn=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.log_fn = log_fn
        self.samples = []

        def log(msg):
            if self.log_fn:
                self.log_fn(str(msg))
            if sys.stdout is not None:
                try:
                    print(msg)
                except Exception:
                    pass

        for label_name, label_idx in self.LABEL_MAP.items():
            folder = self.data_dir / label_name
            if not folder.is_dir():
                log(f"[!] Warning: folder '{folder}' not found - skipping.")
                continue

            for img_path in sorted(folder.iterdir()):
                if img_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    self.samples.append((str(img_path), label_idx))

        if len(self.samples) == 0:
            raise FileNotFoundError(
                f"No images found in '{data_dir}'. Make sure it contains 'good/' and/or 'bad/' subfolders."
            )

        good_count = sum(1 for _, l in self.samples if l == 0)
        bad_count = sum(1 for _, l in self.samples if l == 1)
        log(f"[+] Dataset loaded: {good_count} good, {bad_count} bad ({len(self.samples)} total)")

        self.class_counts = {0: good_count, 1: bad_count}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        roi_image = extract_object_roi(image)
        color_stats = get_color_and_text_stats(roi_image)

        if self.transform:
            spatial_tensor = self.transform(roi_image)
        else:
            spatial_tensor = transforms.ToTensor()(roi_image)

        return spatial_tensor, color_stats, label

    def get_sample_weights(self):
        total = len(self.samples)
        weights = []
        for _, label in self.samples:
            class_count = self.class_counts[label]
            weights.append(total / (2.0 * max(class_count, 1)))
        return weights


# ──────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

predict_transform = val_transform


# ──────────────────────────────────────────────
# Hybrid Model Definition
# ──────────────────────────────────────────────
class HybridClassifier(nn.Module):
    """
    Fuses 512 ResNet spatial & text feature maps with 13 explicit
    color, shade, and text sharpness metrics.
    """

    def __init__(self, num_classes: int = 2, freeze_backbone: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if freeze_backbone else None
        self.resnet = models.resnet18(weights=weights)

        if freeze_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False

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


def build_model(num_classes: int = 2, freeze_backbone: bool = True):
    return HybridClassifier(num_classes=num_classes, freeze_backbone=freeze_backbone)


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
def train_model(data_dir: str = DEFAULT_DATA_DIR, epochs: int = 25,
                batch_size: int = 16, lr: float = 1e-3, val_split: float = 0.2,
                save_path: str = DEFAULT_MODEL_PATH, patience: int = 8,
                log_fn=None):

    def log(msg=""):
        if log_fn:
            log_fn(str(msg))
        if sys.stdout is not None:
            try:
                print(msg)
            except Exception:
                pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"[+] Using device: {device}")
    log(f"[+] Model: {MODEL_ARCH} (ResNet18 + LAB/HSV Color + Text Sharpness)")

    full_dataset = GoodBadDataset(data_dir, transform=train_transform, log_fn=log_fn)
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    val_dataset_for_eval = GoodBadDataset(data_dir, transform=val_transform, log_fn=log_fn)

    sample_weights = full_dataset.get_sample_weights()
    train_weights = [sample_weights[i] for i in train_ds.indices]
    sampler = WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)

    use_pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=0, pin_memory=use_pin_memory)

    val_subset = torch.utils.data.Subset(val_dataset_for_eval, val_ds.indices)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=use_pin_memory)

    log(f"[+] Train samples: {train_size} | Validation samples: {val_size}")

    model = build_model(num_classes=2, freeze_backbone=True).to(device)

    optimizer = optim.Adam(model.classifier.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_val_loss = float("inf")
    epochs_no_improve = 0

    log("-" * 60)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, color_stats, labels in train_loader:
            images = images.to(device)
            color_stats = color_stats.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images, color_stats)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / max(total, 1)
        train_acc = 100.0 * correct / max(total, 1)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, color_stats, labels in val_loader:
                images = images.to(device)
                color_stats = color_stats.to(device)
                labels = labels.to(device)

                outputs = model(images, color_stats)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss = val_loss / max(val_total, 1)
        val_acc = 100.0 * val_correct / max(val_total, 1)

        scheduler.step()

        log(f"  Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:6.2f}% | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:6.2f}%")

        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            epochs_no_improve = 0

            tmp_save_path = save_path + ".tmp"
            torch.save({
                "model_state_dict": model.state_dict(),
                "label_map": {"good": 0, "bad": 1},
                "model_arch": MODEL_ARCH,
                "input_size": INPUT_SIZE,
                "val_acc": val_acc,
                "val_loss": val_loss,
                "epoch": epoch,
            }, tmp_save_path)
            if os.path.exists(save_path):
                try:
                    os.remove(save_path)
                except Exception:
                    pass
            os.replace(tmp_save_path, save_path)
            log(f"  [+] Model saved -> {save_path} (val acc: {val_acc:.2f}%)")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            log(f"\n[-] Early stopping at epoch {epoch}")
            break

    log(f"\n[OK] Training complete! Best validation accuracy: {best_val_acc:.2f}%")
    log(f"     Model saved to: {save_path}")
    return save_path


# ──────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────
def predict_image(image_path: str, model_path: str = DEFAULT_MODEL_PATH):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    label_map = checkpoint.get("label_map", {"good": 0, "bad": 1})
    idx_to_label = {v: k for k, v in label_map.items()}

    model = build_model(num_classes=len(label_map), freeze_backbone=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    raw_image = Image.open(image_path).convert("RGB")
    roi_image = extract_object_roi(raw_image)
    color_stats = get_color_and_text_stats(roi_image).unsqueeze(0).to(device)
    spatial_tensor = predict_transform(roi_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(spatial_tensor, color_stats)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = probabilities.max(1)

    label = idx_to_label[predicted_idx.item()]
    conf = confidence.item() * 100

    print(f"\n[+] Prediction for '{image_path}':")
    print(f"    Label:      {label.upper()}")
    print(f"    Confidence: {conf:.1f}%")
    print(f"    Probabilities: good={probabilities[0][0]:.3f}, bad={probabilities[0][1]:.3f}")

    return label, conf


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Train a Good/Bad image classifier with Color, Text & Background Separation.",
    )
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR,
                        help="Root folder containing 'good/' and 'bad/' subfolders")
    parser.add_argument("--epochs", type=int, default=25,
                        help="Number of training epochs (default: 25)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for classifier head (default: 0.001)")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Fraction of data for validation (default: 0.2)")
    parser.add_argument("--patience", type=int, default=8,
                        help="Early stopping patience in epochs (default: 8)")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH,
                        help="Path to save/load the model (default: model.pth)")
    parser.add_argument("--predict", type=str, default=None,
                        help="Path to a single image to classify (skips training)")

    args = parser.parse_args()

    if args.predict:
        if not os.path.isfile(args.model_path):
            print(f"[-] Model file not found: {args.model_path}")
            sys.exit(1)
        predict_image(args.predict, args.model_path)
    else:
        train_model(
            data_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            val_split=args.val_split,
            save_path=args.model_path,
            patience=args.patience,
        )


if __name__ == "__main__":
    main()
