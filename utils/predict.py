"""
predict.py
----------
Image checks, face detection, skin-tone estimation, and age/gender analysis.
PyTorch inference is optional; the local heuristic is used when unavailable.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

BASE_DIR    = Path(__file__).resolve().parents[1]
MODEL_PATH  = BASE_DIR / "models" / "style_genie_resnet50.pth"
UTKFACE_DIR = BASE_DIR / "UTKFace"

# ── Age buckets (match fashion_dataset.csv) ───────────────────────────────────
AGE_BUCKETS = [
    (0,   4,  "0-4"),
    (5,   9,  "5-9"),
    (10, 14,  "10-14"),
    (15, 19,  "15-19"),
    (20, 24,  "20-24"),
    (25, 29,  "25-29"),
    (30, 34,  "30-34"),
    (35, 39,  "35-39"),
    (40, 44,  "40-44"),
    (45, 49,  "45-49"),
    (50, 54,  "50-54"),
    (55, 99,  "55-59"),
]

SKIN_TONE_SCALE = [
    (55,  float("inf"), "Very Fair"),
    (41,  55,           "Fair"),
    (28,  41,           "Light"),
    (10,  28,           "Light Medium"),
    (-10, 10,           "Medium"),
    (-30, -10,          "Olive"),
    (-50, -30,          "Tan"),
    (-99, -50,          "Brown / Deep"),
]


class FaceAnalysisError(ValueError):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _map_age_group(age: float) -> str:
    age = int(round(float(age)))
    for lo, hi, label in AGE_BUCKETS:
        if lo <= age <= hi:
            return label
    return "55-59"


def _skin_tone_from_face(face_bgr: np.ndarray) -> str:
    h, w = face_bgr.shape[:2]
    roi  = face_bgr[int(h * 0.15): int(h * 0.55),
                    int(w * 0.25): int(w * 0.75)]
    if roi.size == 0:
        roi = face_bgr
    lab    = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    L      = lab[:, :, 0] * (100.0 / 255.0)
    b      = lab[:, :, 2] - 128.0
    b_safe = np.where(np.abs(b) < 1e-3, 1e-3, b)
    ita    = np.degrees(np.arctan((L - 50.0) / b_safe))
    ita_m  = float(np.median(ita))
    for lo, hi, label in SKIN_TONE_SCALE:
        if lo <= ita_m < hi:
            return label
    return "Brown / Deep"


# ─────────────────────────────────────────────────────────────────────────────
# Model definition
# ─────────────────────────────────────────────────────────────────────────────

def _get_transforms():
    import torchvision.transforms as T
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225]),
    ])


def _build_model():
    import torch.nn as nn
    import torchvision.models as models

    backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    in_feats = backbone.fc.in_features
    backbone.fc = nn.Identity()

    class AgeGenderNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone    = backbone
            self.age_head    = nn.Sequential(
                nn.Linear(in_feats, 256), nn.ReLU(), nn.Dropout(0.4),
                nn.Linear(256, 1)
            )
            self.gender_head = nn.Sequential(
                nn.Linear(in_feats, 128), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(128, 1), nn.Sigmoid()
            )

        def forward(self, x):
            f = self.backbone(x)
            return self.age_head(f).squeeze(1), self.gender_head(f).squeeze(1)

    return AgeGenderNet()


# ─────────────────────────────────────────────────────────────────────────────
# UTKFace dataset utilities
# ─────────────────────────────────────────────────────────────────────────────

def _parse_utk_labels(utk_dir: Path) -> list[tuple[str, int, int]]:
    """UTKFace filename: <age>_<gender>_<race>_<timestamp>.jpg  (gender 0=Male, 1=Female)"""
    pattern = re.compile(r'^(\d+)_([01])_\d+_\d+\.jpg', re.IGNORECASE)
    records = []
    for f in utk_dir.glob("*.jpg"):
        m = pattern.match(f.name)
        if m:
            age    = int(m.group(1))
            gender = int(m.group(2))
            if 0 <= age <= 100:
                records.append((str(f), age, gender))
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def _train_and_save(model_path: Path) -> None:
    import random
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image

    print("\n[Style Genie] Training age and gender heads from local UTKFace data")
    print("  The pretrained ResNet backbone is frozen to keep CPU training practical.\n")

    records = _parse_utk_labels(UTKFACE_DIR)
    if len(records) < 100:
        raise RuntimeError(f"UTKFace not found at {UTKFACE_DIR}")

    random.seed(42)
    records = random.sample(records, min(4000, len(records)))

    transform = _get_transforms()

    class UTKDataset(Dataset):
        def __len__(self): return len(records)
        def __getitem__(self, i):
            path, age, gender = records[i]
            img = Image.open(path).convert("RGB")
            # ── KEY FIX: cast to float32, not float64 ──────────────────────
            return transform(img), torch.tensor(age, dtype=torch.float32), \
                                   torch.tensor(gender, dtype=torch.float32)

    loader = DataLoader(UTKDataset(), batch_size=32, shuffle=False, num_workers=0)

    device    = torch.device("cpu")
    model     = _build_model().to(device)
    for parameter in model.backbone.parameters():
        parameter.requires_grad_(False)
    model.eval()

    feature_batches, age_batches, gender_batches = [], [], []
    for i, (imgs, ages, genders) in enumerate(loader):
        with torch.no_grad():
            feature_batches.append(model.backbone(imgs.to(device)))
        age_batches.append(ages)
        gender_batches.append(genders)
        if i % 10 == 0:
            print(f"  Extracting image features: {i + 1}/{len(loader)} batches", flush=True)

    features = torch.cat(feature_batches)
    ages = torch.cat(age_batches)
    genders = torch.cat(gender_batches)
    optimizer = optim.Adam(
        list(model.age_head.parameters()) + list(model.gender_head.parameters()),
        lr=1e-3,
        weight_decay=1e-4,
    )
    age_loss  = nn.L1Loss()
    gen_loss  = nn.BCELoss()

    for epoch in range(12):
        total = 0.0
        indices = torch.randperm(len(features))
        for start in range(0, len(features), 128):
            batch = indices[start:start + 128]
            optimizer.zero_grad()
            pred_age = model.age_head(features[batch]).squeeze(1)
            pred_gen = model.gender_head(features[batch]).squeeze(1)
            loss = age_loss(pred_age, ages[batch]) * 0.05 + gen_loss(pred_gen, genders[batch])
            loss.backward()
            optimizer.step()
            total += loss.item()
        avg = total / ((len(features) + 127) // 128)
        print(f"  Head training epoch {epoch + 1}/12 — loss: {avg:.4f}", flush=True)

    model_path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"\n  [OK] Model saved: {model_path.name}\n")
    logger.info("Model saved to %s", model_path)


# ─────────────────────────────────────────────────────────────────────────────
# Model cache (loaded once per process)
# ─────────────────────────────────────────────────────────────────────────────

_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    import torch

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Trained model checkpoint not found at {MODEL_PATH}; using heuristic fallback."
        )

    model = _build_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()
    _MODEL = model
    logger.info("ResNet50 loaded from %s", MODEL_PATH)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def analyze_face(path: Path) -> dict:
    from PIL import Image

    image = cv2.imread(str(path))
    if image is None:
        raise FaceAnalysisError("Could not read the image file.")
    if min(image.shape[:2]) < 80:
        raise FaceAnalysisError("Image too small — use a photo at least 80×80 px.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if cv2.Laplacian(gray, cv2.CV_64F).var() < 15:
        raise FaceAnalysisError("Photo is too blurry — use a sharper, well-lit image.")

    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    if len(faces) == 0:
        raise FaceAnalysisError(
            "No face detected. Please use a clear front-facing photo in good lighting."
        )

    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    face_bgr   = image[y: y + h, x: x + w]
    skin_tone  = _skin_tone_from_face(face_bgr)

    try:
        import torch

        model     = _get_model()
        transform = _get_transforms()
        face_pil  = Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))
        tensor    = transform(face_pil).unsqueeze(0)

        with torch.no_grad():
            pred_age, pred_gender = model(tensor)

        raw_age    = max(1.0, min(90.0, float(pred_age[0].item())))
        age_group  = _map_age_group(raw_age)

        # UTKFace: 0 = Male, 1 = Female
        gender_prob = float(pred_gender[0].item())
        gender      = "Female" if gender_prob >= 0.5 else "Male"
        gender_conf = round(max(0.55, min(0.97,
                        gender_prob if gender == "Female" else 1.0 - gender_prob)), 2)
        age_conf    = round(min(0.92, max(0.62,
                        0.90 - abs(raw_age % 5 - 2.5) * 0.03)), 2)
        model_tag   = "ResNet50 on UTKFace (PyTorch)"

    except Exception as exc:
        logger.warning("PyTorch inference error: %s", exc, exc_info=True)
        age_group, gender, age_conf, gender_conf = _heuristic(face_bgr)
        model_tag = "No trained age/gender model available"

    return {
        "age_group":  age_group,
        "gender":     gender,
        "skin_tone":  skin_tone,
        "confidence": {
            "age":       age_conf,
            "gender":    gender_conf,
            "skin_tone": 0.80,
        },
        "face_box":   {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
        "model":      model_tag,
        "needs_input": age_group == "Needs input" or gender == "Needs input",
    }


def _heuristic(face_bgr: np.ndarray) -> tuple:
    """Do not invent age/gender values when the trained model is unavailable."""
    return "Needs input", "Needs input", 0.0, 0.0
