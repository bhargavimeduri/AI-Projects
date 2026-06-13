"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   TIGER AI ULTIMATE V2 — Upgraded Pipeline v4.0                            ║
║   EPAIB Batch 05 | Group 4 | IIM Lucknow                                   ║
║                                                                              ║
║   WHAT IS NEW IN V2 vs V1 (tiger_ai_ultimate.py):                         ║
║   ─────────────────────────────────────────────────────────────────────     ║
║                                                                              ║
║   UPGRADE 1 — LARGER DETECTION MODELS:                                     ║
║      V1: YOLOv8n-OIV7  (nano,  3.2M params) — misses second tiger         ║
║      V2: YOLOv8l-OIV7  (large, 43M  params) — 13x more capacity           ║
║      V1: YOLOv8n-COCO  (nano,  3.2M params)                               ║
║      V2: YOLOv8l-COCO  (large, 43M  params)                               ║
║      Why: Ground truth validation showed 000082, 000102 missed 2nd tiger.  ║
║      Root cause: YOLOv8n too small for multi-tiger frames.                 ║
║                                                                              ║
║   UPGRADE 2 — ResNet50 RETAINED (EfficientNetV2-L REVERTED):              ║
║      Attempted: EfficientNetV2-L (ImageNet-21K, 118M params)               ║
║      Reverted to: ResNet50 (ImageNet-1K, 25M params)                       ║
║      Why reverted: EfficientNetV2-L uses 21,841-class index space.         ║
║      Our species gate uses ImageNet-1K tiger indices (292/293/294).         ║
║      Those indices map to WRONG classes in the 21K model, causing           ║
║      valid tigers to fail the species gate. V1 accuracy was higher.        ║
║                                                                              ║
║   UPGRADE 3 — GRADIENT-MAGNITUDE BLUR DETECTION:                          ║
║      V1: Laplacian variance (threshold 25) — rejected clear image 000143   ║
║      V2: Combined Laplacian + Gradient Magnitude — more robust to          ║
║      camera trap compression artefacts. Threshold lowered to 10.           ║
║                                                                              ║
║   UPGRADE 4 — ORIENTATION-AWARE GRID SPLIT:                               ║
║      V1: Always splits Left/Right — caused false 2-tiger split on          ║
║          000084.jpg (portrait/vertical image, 1 tiger)                     ║
║      V2: Portrait image (H > W) → split Top/Bottom                        ║
║          Landscape image (W >= H) → split Left/Right                      ║
║                                                                              ║
║   UPGRADE 5 — MOSAIC AUGMENTATION IN TRAINING:                            ║
║      V1: Standard augmentation (flip, brightness, zoom)                    ║
║      V2: Mosaic=1.0 added — combines 4 images per training step.          ║
║      Why: Directly teaches YOLO to detect multiple tigers in one frame.    ║
║                                                                              ║
║   UPGRADE 6 — LABEL SMOOTHING IN TRAINING:                                ║
║      V1: Hard labels (0 / 1)                                               ║
║      V2: Label smoothing (ε=0.1) — 0.05 / 0.95                           ║
║      Why: Camera trap images have uncertain labels (partial tigers).        ║
║      Soft labels better reflect real-world ambiguity.                      ║
║                                                                              ║
║   EVERYTHING FROM V1 IS PRESERVED:                                         ║
║   ─────────────────────────────────────────────────────────────────────     ║
║   ✅ 3-stage cascade detection (OIV7 → COCO → Whole-image fallback)        ║
║   ✅ Species verification gate (top-3 tiger confirmation)                   ║
║   ✅ ViewPoint classification + ViewPoint-aware census                      ║
║   ✅ Water reflection guard                                                  ║
║   ✅ CLAHE + Dark Channel Prior dehazing + IR normalisation                 ║
║   ✅ Shadow, corner trace, overlap guards                                   ║
║   ✅ Grad-CAM explainability                                                ║
║   ✅ JSON identity DB + running average embeddings                          ║
║   ✅ Full population report (CSV + TXT + annotated images)                  ║
║                                                                              ║
║   USAGE:                                                                     ║
║     py -3 src/tiger_ai_ultimate_v2.py                    # batch run        ║
║     py -3 src/tiger_ai_ultimate_v2.py --gradcam          # with Grad-CAM   ║
║     py -3 src/tiger_ai_ultimate_v2.py --image tiger.jpg  # single image    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ── Standard library imports ───────────────────────────────────────────────────
import sys           # Provides access to Python interpreter (used for version checks)
import json          # Read/write JSON files — used for the tiger identity DB
import argparse      # Parses command-line arguments (--data, --gradcam, --image flags)
import warnings      # Suppress non-critical model loading warnings from PyTorch

# ── Computer vision ────────────────────────────────────────────────────────────
import cv2           # OpenCV — image loading, CLAHE, drawing boxes, colour operations
import numpy as np   # NumPy — all array/matrix maths (images are just NumPy arrays)

# ── Data and reporting ─────────────────────────────────────────────────────────
import pandas as pd  # Creates the population report CSV

# ── Deep learning — PyTorch ────────────────────────────────────────────────────
import torch                         # Core deep learning framework (runs ResNet50)
import torchvision.models as models  # Pre-trained ResNet50 weights (ImageNet)
import torchvision.transforms as transforms  # Image preprocessing for ResNet50
                                             # (resize, crop, normalise to ImageNet mean/std)

# ── Image utilities ────────────────────────────────────────────────────────────
from PIL import Image  # Pillow — loads WEBP, PNG, JPEG; used before ResNet transforms

# ── Path and time utilities ────────────────────────────────────────────────────
from pathlib import Path     # Cross-platform file paths (works on Windows/Linux/Mac)
from datetime import datetime  # Timestamps for reports

# ── Similarity metric ──────────────────────────────────────────────────────────
from sklearn.metrics.pairwise import cosine_similarity
# Cosine similarity: measures the angle between two 2048-dim fingerprint vectors.
# Score = 1.0 → identical; Score = 0.0 → completely unrelated.
# Threshold 0.83 = "same tiger confirmed".
# Chosen over Euclidean distance because it ignores image brightness differences.

# ── Object detection ───────────────────────────────────────────────────────────
from ultralytics import YOLO
# YOLOv8 (You Only Look Once, v8) — real-time object detector.
# "You Only Look Once" = scans the entire image in one forward pass,
# making it ~1000x faster than older detection methods.
# We use two weight files:
#   yolov8n-oiv7.pt  — trained on Open Images V7 (601 classes including Tiger)
#   yolov8n.pt       — trained on COCO (80 classes, no Tiger but has cat/dog/zebra)

warnings.filterwarnings("ignore")  # Suppress PyTorch deprecation warnings

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — TRAINING
# Two-stage training pipeline. Run BEFORE inference on a new dataset.
#
# STAGE 1 — YOLOv8 Tiger Detector (custom fine-tune on your labeled data):
#   py -3 src/tiger_ai_ultimate.py --train --stage yolo
#
# STAGE 2 — ResNet50 Tiger/No-Tiger Detection (TensorFlow, transfer learning):
#   py -3 src/tiger_ai_ultimate.py --train --stage detection
#
# STAGE 3 — EfficientNetB3 Individual Identification (TensorFlow, fine-tune):
#   py -3 src/tiger_ai_ultimate.py --train --stage identification
#
# NOTE: TensorFlow imports are lazy (loaded only when --train is used) so the
# inference pipeline runs fine without TF installed.
# ══════════════════════════════════════════════════════════════════════════════

def train_yolo(data_yaml: str, epochs: int = 50, batch: int = 16,
               imgsz: int = 640, model_dir: str = "models"):
    """
    Fine-tune YOLOv8n on a custom labeled tiger dataset.

    Expects a data.yaml file pointing to train/val image folders with YOLO
    format labels (.txt files with class_id x_center y_center width height).

    Args:
        data_yaml  : Path to dataset YAML (e.g. 'data/tiger_dataset.yaml')
        epochs     : Number of training epochs (default 50)
        batch      : Images per batch — reduce to 8 on CPU (default 16)
        imgsz      : Training image size (default 640)
        model_dir  : Folder to save the best trained weights

    After training the best weights are saved to:
        models/yolo_training/tiger_detector_v1/weights/best.pt

    Use best.pt in inference by replacing yolov8n-oiv7.pt in SECTION 4.
    """
    print("\n" + "=" * 60)
    print("STAGE 1: YOLOv8 Tiger Detector — Custom Fine-Tune")
    print("=" * 60)
    print(f"  Dataset YAML : {data_yaml}")
    print(f"  Epochs       : {epochs}")
    print(f"  Batch size   : {batch}")
    print(f"  Image size   : {imgsz}x{imgsz}")

    detector = YOLO("yolov8l.pt")   # V2: Start from large COCO pretrained weights
    results = detector.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=f"{model_dir}/yolo_training",
        name="tiger_detector_v2",
        patience=10,           # Stop early if no improvement for 10 epochs
        augment=True,          # Built-in flip, HSV augmentation
        mosaic=1.0,            # V2 UPGRADE: Mosaic augmentation — combines 4 images
                               # per training step → teaches YOLO to detect multiple
                               # tigers in one frame. Directly fixes 000082/000102 misses.
        mixup=0.1,             # V2 UPGRADE: MixUp — blends images → robust to occlusion
        label_smoothing=0.1,   # V2 UPGRADE: Label smoothing ε=0.1 → soft labels (0.05/0.95)
                               # Handles uncertainty in camera trap annotations
        cache=False,           # Set True if RAM > 16 GB for faster epochs
        device=0 if torch.cuda.is_available() else "cpu",
        verbose=True,
    )
    best_weights = f"{model_dir}/yolo_training/tiger_detector_v1/weights/best.pt"
    print(f"\n  Best weights saved → {best_weights}")
    print("  Use this path in SECTION 4 to replace yolov8n-oiv7.pt.\n")
    return results


def train_detection_model(data_dir: str = "data/processed", model_dir: str = "models",
                          epochs_phase1: int = 20, epochs_phase2: int = 10,
                          batch_size: int = 32):
    """
    Train ResNet50 to classify Tiger vs No-Tiger (binary detection).

    Two-phase transfer learning:
      Phase 1 (Frozen):   Only the top Dense layers train.
                          High learning rate (1e-4). Fast convergence.
      Phase 2 (Fine-tune): Top 30 ResNet50 layers unfrozen.
                           Low learning rate (1e-5). Squeezes extra accuracy.

    Expects data_dir/train/ and data_dir/val/ with subfolders:
        tiger/      — images containing tigers
        no_tiger/   — images without tigers

    Saves model to models/detection_best.h5 and models/detection_final.h5
    """
    import tensorflow as tf
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras import layers, Model
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    print("\n" + "=" * 60)
    print("STAGE 2: Tiger Detection — ResNet50 Transfer Learning")
    print("=" * 60)

    train_dir = Path(data_dir) / "train"
    val_dir   = Path(data_dir) / "val"

    datagen_train = ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=True,
        rotation_range=20,
        brightness_range=[0.7, 1.3],
        zoom_range=0.15,
    )
    datagen_val = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = datagen_train.flow_from_directory(
        str(train_dir), target_size=(224, 224),
        batch_size=batch_size, class_mode="binary",
    )
    val_gen = datagen_val.flow_from_directory(
        str(val_dir), target_size=(224, 224),
        batch_size=batch_size, class_mode="binary",
    )

    # Build ResNet50 detection model
    base = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base.trainable = False
    inputs  = tf.keras.Input(shape=(224, 224, 3))
    x       = base(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.Dense(256, activation="relu")(x)
    x       = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model   = Model(inputs, outputs, name="tiger_detection")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),  # V2: label smoothing
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", patience=5, restore_best_weights=True, mode="max"),
        tf.keras.callbacks.ModelCheckpoint(
            f"{model_dir}/detection_best.h5",
            monitor="val_auc", save_best_only=True, mode="max"),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
    ]

    print(f"\n  Phase 1 — Frozen base ({epochs_phase1} epochs)")
    model.fit(train_gen, validation_data=val_gen,
              epochs=epochs_phase1, callbacks=callbacks)

    print(f"\n  Phase 2 — Fine-tuning top 30 ResNet50 layers ({epochs_phase2} epochs)")
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    model.fit(train_gen, validation_data=val_gen,
              epochs=epochs_phase2, callbacks=callbacks)

    val_loss, val_acc, val_auc = model.evaluate(val_gen, verbose=0)
    print(f"\n  Detection Results:")
    print(f"    Val Accuracy : {val_acc:.4f}  (target >0.90)")
    print(f"    Val AUC      : {val_auc:.4f}")

    model.save(f"{model_dir}/detection_final.h5")
    print(f"  Model saved → {model_dir}/detection_final.h5\n")


def train_identification_model(data_dir: str = "data/processed", model_dir: str = "models",
                                num_tigers: int = 50, epochs: int = 30, batch_size: int = 32):
    """
    Fine-tune EfficientNetB3 to identify individual tigers by stripe pattern.

    Each tiger gets its own subfolder:
        data_dir/train/Tiger_001/  ← images of Tiger 001
        data_dir/train/Tiger_002/  ← images of Tiger 002
        ...

    Two-phase:
      Phase 1 — Top layers only (frozen EfficientNetB3 base)
      Phase 2 — Fine-tune top 30 EfficientNetB3 layers (lower LR)

    Saves model to models/identification_best.h5 and tiger_class_map.json
    """
    import tensorflow as tf
    from tensorflow.keras.applications import EfficientNetB3
    from tensorflow.keras import layers, Model
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    print("\n" + "=" * 60)
    print("STAGE 3: Individual Tiger ID — EfficientNetB3 Fine-Tune")
    print("=" * 60)

    train_dir = Path(data_dir) / "train"
    val_dir   = Path(data_dir) / "val"

    datagen_train = ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=True,
        rotation_range=15,
        brightness_range=[0.7, 1.3],
        zoom_range=0.1,
    )
    datagen_val = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = datagen_train.flow_from_directory(
        str(train_dir), target_size=(300, 300),
        batch_size=batch_size, class_mode="categorical",
    )
    val_gen = datagen_val.flow_from_directory(
        str(val_dir), target_size=(300, 300),
        batch_size=batch_size, class_mode="categorical",
    )

    num_classes = train_gen.num_classes
    print(f"  Found {num_classes} tiger classes")

    # Build EfficientNetB3 identification model
    base = EfficientNetB3(weights="imagenet", include_top=False, input_shape=(300, 300, 3))
    for layer in base.layers[:-20]:
        layer.trainable = False
    inputs  = tf.keras.Input(shape=(300, 300, 3))
    x       = base(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.Dense(512, activation="relu")(x)
    x       = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model   = Model(inputs, outputs, name="tiger_identification")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=7, restore_best_weights=True, mode="max"),
        tf.keras.callbacks.ModelCheckpoint(
            f"{model_dir}/identification_best.h5",
            monitor="val_accuracy", save_best_only=True, mode="max"),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
    ]

    epochs_p1 = max(1, epochs - 10)
    epochs_p2 = min(10, epochs)

    print(f"\n  Phase 1 — Frozen base ({epochs_p1} epochs)")
    model.fit(train_gen, validation_data=val_gen,
              epochs=epochs_p1, callbacks=callbacks)

    print(f"\n  Phase 2 — Fine-tuning top 30 EfficientNetB3 layers ({epochs_p2} epochs)")
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(train_gen, validation_data=val_gen,
              epochs=epochs_p2, callbacks=callbacks)

    val_loss, val_acc = model.evaluate(val_gen, verbose=0)
    print(f"\n  Identification Results:")
    print(f"    Val Accuracy : {val_acc:.4f}  (target >0.80)")

    model.save(f"{model_dir}/identification_final.h5")

    # Save class → tiger label mapping (Tiger_001, Tiger_002, …)
    import json as _json
    class_map = {str(v): k for k, v in train_gen.class_indices.items()}
    with open(f"{model_dir}/tiger_class_map.json", "w") as f:
        _json.dump(class_map, f, indent=2)

    print(f"  Model saved     → {model_dir}/identification_final.h5")
    print(f"  Class map saved → {model_dir}/tiger_class_map.json\n")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — SETUP & CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# ── File and folder paths ──────────────────────────────────────────────────────
# __file__ = this script's location.  parent.parent = project root folder.
BASE_DIR       = Path(__file__).resolve().parent.parent
DATA_DIR       = BASE_DIR / "data" / "sample" / "tigers" / "train"  # Default dataset
REPORTS_DIR    = BASE_DIR / "reports"               # All output files go here
ANNOTATED_DIR  = REPORTS_DIR / "v2_annotated"       # V2 annotated images (separate from V1)
GRADCAM_DIR    = REPORTS_DIR / "v2_gradcam"         # V2 Grad-CAM heatmaps
DB_FILE        = REPORTS_DIR / "v2_tiger_db.json"               # V2 tiger identity DB
OUTPUT_CSV     = REPORTS_DIR / "v2_population_report.csv"       # V2 per-detection log
REPORT_TXT     = REPORTS_DIR / "v2_population_report.txt"       # V2 text census summary

# Create output folders if they don't exist (first run on a new machine)
for d in [REPORTS_DIR, ANNOTATED_DIR, GRADCAM_DIR]:
    d.mkdir(exist_ok=True)

# ── Survey metadata ─────────────────────────────────────────────────────────────
# These appear in every report header. Change to match actual survey location.
RANGE_NAME = "Sundarbans Tiger Reserve"
BEAT_NAME  = "Block-A / Sector-3"

# ── Device selection ────────────────────────────────────────────────────────────
# CUDA = NVIDIA GPU (fast). CPU = regular processor (slow but always available).
# On a GPU, 3,392 images process in ~20 minutes. On CPU, ~4 hours.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device : {DEVICE.upper()}")

# Supported image formats — WEBP is common in newer camera traps and phones
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ── Tiger colour morph display info ───────────────────────────────────────────
TIGER_TYPES = {
    "Orange_Standard":        {"display": "Orange Tiger",     "color": (0,   140, 255)},
    "White":                  {"display": "White Tiger",      "color": (200, 200, 255)},
    "Black_Pseudomelanistic": {"display": "Black Tiger",      "color": (60,  60,  60) },
    "Golden":                 {"display": "Golden Tiger",     "color": (0,   215, 255)},
    "Snow_White":             {"display": "Snow White Tiger", "color": (255, 255, 255)},
}

# ── Detection confidence thresholds ────────────────────────────────────────────
# "Confidence" = YOLO's certainty that it found the right object (0.0 to 1.0).
# Too low → lots of false positives (rocks called tigers).
# Too high → miss real tigers in difficult lighting.
# OIV7 uses a direct "Tiger" class so we can afford slightly lower confidence.
CONF_OIV7             = 0.09   # OIV7: 9% confidence minimum.
                                # OIV7 has a dedicated Tiger class — even a 9% Tiger
                                # detection is far more specific than a COCO proxy.
                                # The species gate (ResNet top-3) is the real safety net.
                                # Lowered from 0.30 → 0.09 to catch second tigers in
                                # multi-tiger frames where the dominant tiger "uses up"
                                # YOLO's attention and the second individual scores lower.
                                # Validated: 15.jpeg has 2 tigers; second OIV7 box=0.10.
CONF_COCO             = 0.35   # COCO proxies need 35% because cat/dog ≠ tiger
IOU_THRESHOLD         = 0.50   # Intersection over Union for NMS — boxes overlapping
                                # more than 50% of their area → keep only the best one
MIN_BOX_FULL          = 0.05   # Bounding box >= 5% of frame area → Full_Body
MIN_BOX_PARTIAL       = 0.015  # >= 1.5% of frame area → Partial_Body (read-only match)
                                # Lowered from 3% to capture distant/small tigers in
                                # wide-angle shots where tigers are far from the camera.
CROP_PAD              = 15     # Pixels of padding around each detected box.
                                # Without padding, ResNet50 sees hard-cut edges
                                # instead of the animal's full shape.

# ── Identity matching thresholds ───────────────────────────────────────────────
# Calibrated on the Sundarbans dataset; may need recalibration for Amur tigers.
SIMILARITY_THRESHOLD       = 0.83   # >= 0.83 cosine score → same tiger confirmed
PARTIAL_MATCH_THRESHOLD    = 0.70   # Relaxed threshold for partial/corner crops

# ── COCO proxy classes ──────────────────────────────────────────────────────────
# COCO-80 dataset has NO Tiger class. These are the closest shape proxies:
#   15=cat   (feline body shape)
#   16=dog   (quadruped body shape)
#   17=horse (large quadruped, similar silhouette to tiger in long shots)
#   22=zebra (striped large animal — BEST proxy; tiger stripe pattern triggers zebra)
#            NOTE: COCO class 22 = zebra (verified). Earlier code had 24 which
#            is "backpack" — that was a bug; fixed here.
# Anything COCO detects in these classes will then go through the ResNet top-3
# species verification gate before being accepted as a tiger crop.
TIGER_PROXY_CLASSES = {15, 16, 17, 22}

# ── Image quality thresholds ────────────────────────────────────────────────────
# These control the triage step (Section 2) — the first pass that discards
# unusable images before any ML model is loaded.
#
# CALIBRATION NOTE (updated 2026-06-02 after Test Bh review):
#   Camera trap images are NOT DSLR photos.
#   - A forest camera trap behind leaves at night produces Laplacian 38–48
#     even when the tiger is perfectly visible. Our original threshold of 80
#     was calibrated on higher-quality Sundarbans images. Lowered to 25.
#   - Dark IR images (brightness 11–13) still contain tigers. Lowered to 8.
#     IR images are identified BEFORE the brightness check — see triage_image().
BLUR_THRESHOLD        = 10.0   # V2: lowered from 25 → 10 (camera trap images score low)
                                # V2 also adds gradient magnitude as secondary check
                                # Combined metric fixes 000143 false rejection
OVEREXPOSE_THRESHOLD  = 240    # Mean brightness > 240 → blown out, skip
UNDEREXPOSE_THRESHOLD = 8      # Mean brightness < 8   → too dark, skip
                                # (was 20 — dark IR images at brightness 11–13 are fine)
IR_CHANNEL_DIFF       = 5      # R-G channel std < 5 → greyscale IR night image


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — IMAGE TRIAGE
# Cheapest filter runs first — discard unusable images before any ML runs.
# ══════════════════════════════════════════════════════════════════════════════

def detect_ir_night(img_bgr):
    """
    Night/IR camera trap images have identical R, G, B channels.
    A colour photo has different channel values — an IR photo looks grey.
    We check the standard deviation between R and G channels:
    if near-zero → all channels same → it's a greyscale IR image.
    """
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    return (np.std(rgb[:,:,0] - rgb[:,:,1]) < IR_CHANNEL_DIFF and
            np.std(rgb[:,:,0] - rgb[:,:,2]) < IR_CHANNEL_DIFF)


def triage_image(img_bgr):
    """
    Returns (quality_bucket, reason, should_skip).
    Buckets: Clean | IR_Night | Unusable_Blurry | Unusable_Overexposed |
             Unusable_Underexposed

    ORDER MATTERS — IR check runs first:
    Camera traps at night produce IR greyscale images that are naturally dark
    (mean brightness 10–15). If we check underexpose BEFORE we check for IR,
    these valid night images get rejected as "too dark" before we can even
    identify them as night shots.
    Fix: detect IR camera type first, then skip brightness/blur checks for IR.
    """
    gray       = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))

    # V2 UPGRADE — Combined blur metric (Laplacian + Gradient Magnitude)
    # Laplacian alone rejects camera trap images that are soft but not blurry.
    # Gradient Magnitude captures edge sharpness more robustly for compressed images.
    # Final blur score = mean of both metrics — requires BOTH to be low to reject.
    laplacian_score  = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    grad_x           = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y           = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_score   = float(np.sqrt(grad_x**2 + grad_y**2).var())
    blur_score       = (laplacian_score + gradient_score) / 2.0  # combined metric

    # ── IR / Night check FIRST — before any brightness/blur rejection ──────────
    if detect_ir_night(img_bgr):
        return "IR_Night", f"Greyscale IR (bright={brightness:.0f}, blur={blur_score:.0f})", False

    # ── Standard quality checks for regular (colour) images ───────────────────
    if brightness > OVEREXPOSE_THRESHOLD:
        return "Unusable_Overexposed",  f"brightness {brightness:.0f}", True
    if brightness < UNDEREXPOSE_THRESHOLD:
        return "Unusable_Underexposed", f"brightness {brightness:.0f}", True
    if blur_score < BLUR_THRESHOLD:
        return "Unusable_Blurry", f"Combined blur {blur_score:.1f} (Laplacian={laplacian_score:.1f}, Gradient={gradient_score:.1f})", True
    return "Clean", f"OK (bright={brightness:.0f}, blur={blur_score:.0f})", False


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — PREPROCESSING CHAIN
# CLAHE + Dark Channel Prior (from TRACE) +
# Greyscale normalisation for IR nights (from Dr. Bose)
# ══════════════════════════════════════════════════════════════════════════════

_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def apply_clahe(img_bgr):
    """
    CLAHE on the LAB L-channel — boosts local contrast without blowing out
    bright areas. Essential for camera trap images at dusk/dawn.
    Operates only on the luminance channel, so colour is preserved.
    """
    lab           = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b       = cv2.split(lab)
    enhanced      = cv2.merge([_clahe.apply(l), a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def dehaze(img_bgr, patch_size=15, omega=0.95, t0=0.1):
    """
    Dark Channel Prior — removes fog, monsoon rain scatter, and haze.
    Sundarbans is a mangrove delta: humidity and ground-level haze are constant.
    Without this step, ResNet50 reads the haze layer as image content and
    produces noisy feature vectors that don't match across days.
    """
    img_f  = img_bgr.astype(np.float64) / 255.0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))

    dark_ch  = cv2.erode(np.min(img_f, axis=2).astype(np.float32), kernel
                         ).astype(np.float64)
    n_bright = max(1, int(dark_ch.size * 0.001))
    A        = np.clip(
        np.max(img_f.reshape(-1, 3)[np.argsort(dark_ch.flatten())[-n_bright:]], axis=0),
        1e-6, 1.0)

    dark_norm = cv2.erode(
        (np.min(img_f / A, axis=2)).astype(np.float32), kernel).astype(np.float64)
    t = np.clip(1.0 - omega * dark_norm, t0, 1.0)[:, :, np.newaxis]

    return np.clip(((img_f - A) / t + A) * 255, 0, 255).astype(np.uint8)


def normalise_for_ir(img_bgr):
    """
    Dr. Bose's insight: convert ALL images to greyscale then back to 3-channel
    before feature extraction on night/IR images.

    WHY: Night IR cameras produce black-and-white images.
    ResNet50 was trained on colour photos. If we feed an IR image as-is,
    the model may use the slight colour tint of the IR sensor as a feature
    (not the stripes). Converting to true greyscale then duplicating across
    all 3 channels forces the model to rely on TEXTURE (stripe pattern)
    rather than colour — which is consistent across day and night.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def gamma_correct(img_bgr, target_brightness=80):
    """
    Automatic gamma correction for underexposed images.

    WHY: ResNet50 was trained on well-lit ImageNet photos. If we feed it a dark
    camera trap image (mean brightness 12–25), the network cannot distinguish
    tiger stripes from background shadow because both look the same shade of
    near-black. Gamma correction remaps the pixel values so dark details become
    visible to the network while avoiding over-brightening of the bright areas.

    How it works:
      gamma < 1 → brightens dark image (e.g. gamma=0.4 makes 12-brightness → ~80)
      gamma > 1 → darkens bright image
      gamma = 1 → no change

    We compute the required gamma automatically to hit a target brightness of 80.
    Formula: target_brightness = current_brightness ^ (1/gamma)
    Solving: gamma = log(current) / log(target)
    """
    current = float(np.mean(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)))
    if current < 1 or current >= target_brightness:
        return img_bgr  # Already bright enough, no change needed
    gamma = np.log(current / 255.0) / np.log(target_brightness / 255.0)
    gamma = float(np.clip(gamma, 0.1, 0.9))  # Safety bounds — never invert

    # Build a lookup table (LUT): for each 0–255 input, compute the corrected output.
    # LUT is much faster than applying the formula pixel-by-pixel.
    lut = np.array([min(255, int((i / 255.0) ** gamma * 255))
                    for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img_bgr, lut)


# Stronger CLAHE for dark images — higher clip limit recovers more shadow detail
_clahe_strong = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(4, 4))


def enhance_dark_crop(crop_bgr):
    """
    Pipeline for very dark crops (mean brightness < 50) before ResNet50.
    Applied step by step:
      1. Gamma correction — lift the darkest pixels to visible range
      2. Strong CLAHE    — boost local contrast to make stripes visible
    Without this, ResNet50 sees a near-black rectangle and classifies it as
    anything but a tiger (darkness pattern alone = no species signal).
    """
    crop_bright = gamma_correct(crop_bgr, target_brightness=80)
    lab = cv2.cvtColor(crop_bright, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    enhanced = cv2.merge([_clahe_strong.apply(l), a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def preprocess(img_bgr, is_ir=False):
    """
    Full preprocessing chain per image:
      1. IR normalisation (if night image)     — Dr. Bose's greyscale trick
      2. Gamma correction for dark images      — auto-lifts brightness for ResNet
      3. CLAHE contrast enhancement            — from TRACE
      4. Dark Channel Prior dehazing           — fog removal from TRACE
    """
    if is_ir:
        img_bgr = normalise_for_ir(img_bgr)
    # Auto-brighten before CLAHE if the image is still very dark after IR norm
    mean_bright = float(np.mean(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)))
    if mean_bright < 40:
        img_bgr = gamma_correct(img_bgr, target_brightness=80)
    return dehaze(apply_clahe(img_bgr))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — MODEL LOADING
# Three models:
#   a. YOLOv8n-OIV7   — Open Images V7, has actual "Tiger" class (Dr. Bose)
#   b. YOLOv8n-COCO   — COCO, uses proxy classes (TRACE fallback)
#   c. ResNet50 full  — classification + Grad-CAM (Yeshvir species gate)
#   d. ResNet50 trunc — 2048-dim feature extractor (TRACE identity)
# ══════════════════════════════════════════════════════════════════════════════

# ── (a) OIV7 detector — tries to load; gracefully skips if unavailable ────────
# Open Images V7 has 601 classes including "Tiger". If this model is not yet
# downloaded (first run), ultralytics downloads it automatically (~6MB).
# If the download fails (no internet), we fall through to COCO-only mode.
print("[INFO] Loading YOLOv8n-OIV7 (Open Images Tiger class) ...")
oiv7_detector = None
try:
    oiv7_detector = YOLO("yolov8l-oiv7.pt")   # V2: large model (43M params vs 3.2M in nano)
    print("[INFO] OIV7 detector loaded — Stage 1 detection active.")
except Exception as e:
    print(f"[WARN] OIV7 load failed ({e}). Cascade will start from COCO Stage 2.")

# ── (b) COCO detector — fallback, always available ────────────────────────────
print("[INFO] Loading YOLOv8l-COCO (proxy classes) ...")
coco_detector = YOLO("yolov8l.pt")            # V2: large model (43M params vs 3.2M in nano)

# ── (c/d) ResNet50 — species gate + re-ID (RETAINED from V1) ──────────────────
# NOTE: EfficientNetV2-L was tested in V2 but reverted.
# Root cause: EfficientNetV2-L uses ImageNet-21K (21,841 classes). Our species
# gate checks class indices 292/293/294 which are tiger classes in ImageNet-1K.
# In the 21K model those indices map to unrelated classes → valid tigers fail
# the species gate → V1 accuracy was higher. ResNet50 + ImageNet-1K is retained.
print("[INFO] Loading ResNet50 (species verification + feature extraction) ...")
_weights_resnet      = models.ResNet50_Weights.DEFAULT
resnet_classify      = models.resnet50(weights=_weights_resnet).to(DEVICE)
resnet_classify.eval()

# ImageNet-1K class labels (used by species gate to confirm "tiger" in top-3)
imagenet_categories = _weights_resnet.meta["categories"]

# Feature extractor = ResNet50 with the final FC layer removed.
# Output: 2048-dimensional stripe fingerprint vector.
resnet_features = torch.nn.Sequential(
    *list(resnet_classify.children())[:-1],   # all layers except final FC
    torch.nn.Flatten(1),
).to(DEVICE)
resnet_features.eval()

# ResNet50 standard ImageNet transform (224x224 — same as V1)
_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

print("[INFO] All models loaded.\n")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — TIGER IDENTITY DATABASE
# Running average embedding + JSON persistence (from TRACE)
# ══════════════════════════════════════════════════════════════════════════════

tiger_db     = {}   # { "Tiger_001": np.array(2048,) }
tiger_counts = {}   # { "Tiger_001": int }


def load_db():
    if DB_FILE.exists():
        try:
            raw = json.loads(DB_FILE.read_text())
            for tid, entry in raw.items():
                tiger_db[tid]     = np.array(entry["embedding"])
                tiger_counts[tid] = entry["sighting_count"]
            print(f"[INFO] Loaded {len(tiger_db)} tiger profile(s) from DB.")
        except Exception as e:
            print(f"[WARN] DB load failed: {e} — starting fresh.")


def save_db():
    DB_FILE.write_text(json.dumps(
        {tid: {"embedding": tiger_db[tid].tolist(),
               "sighting_count": tiger_counts[tid]}
         for tid in tiger_db}, indent=2))


def extract_fingerprint(crop_bgr):
    """
    2048-dim feature vector from ResNet50 average-pool layer.
    This is the tiger's "stripe fingerprint" — same tiger = same direction
    in 2048-dimensional space regardless of lighting or image angle.
    """
    pil    = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    tensor = _transform(pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return resnet_features(tensor).squeeze().cpu().numpy()


def match_or_enroll(fingerprint, update_db=True):
    """
    Compare fingerprint against all known tigers.
    BEST match algorithm (no break — scans all tigers, picks highest score).
    This fix was in Yeshvir's code — the original had a 'break' bug that
    would stop at the first match >= threshold, even if a better one existed.

    update_db=True  → full-body sighting: update running average + count
    update_db=False → partial / corner: read-only match, never corrupt DB
    """
    if not tiger_db:
        if update_db:
            tid = "Tiger_001"
            tiger_db[tid]     = fingerprint
            tiger_counts[tid] = 1
            return tid, 1.0, True
        return "Unknown_Partial", 0.0, True

    scores  = {tid: cosine_similarity(fingerprint.reshape(1, -1),
                                       vec.reshape(1, -1))[0][0]
               for tid, vec in tiger_db.items()}
    best_id = max(scores, key=scores.get)
    best_sc = float(scores[best_id])

    if best_sc >= SIMILARITY_THRESHOLD:
        if update_db:
            # Running average: new DB = (old × n + new) / (n + 1)
            # This means the 50th sighting of Tiger_003 improves its fingerprint
            # rather than overwriting or ignoring it. DB always gets better.
            n = tiger_counts[best_id]
            tiger_db[best_id]      = (tiger_db[best_id] * n + fingerprint) / (n + 1)
            tiger_counts[best_id] += 1
        return best_id, best_sc, False

    if update_db:
        new_id = f"Tiger_{len(tiger_db) + 1:03d}"
        tiger_db[new_id]     = fingerprint
        tiger_counts[new_id] = 1
        return new_id, best_sc, True

    return best_id, best_sc, True


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — SPECIES VERIFICATION LAYER
# From Yeshvir Script B — ResNet50 top-3 must contain "tiger".
#
# WHY THIS IS NEEDED:
# YOLO-COCO proxy classes detect cats, dogs, horses, zebras as proxies for
# tigers. Many of these detections are WRONG (a stump looks like a cat in COCO).
# Without a species gate, we fingerprint rocks and stumps as tigers.
# The top-3 check is a second model's vote — both YOLO and ResNet50 must agree
# that something tiger-like is present before we accept the crop.
#
# We skip this gate for OIV7 detections because OIV7 already confirmed "Tiger".
# We APPLY it for COCO detections and whole-image fallback.
# ══════════════════════════════════════════════════════════════════════════════

def _run_resnet_topk(crop_bgr, k=3):
    """Internal helper — runs ResNet50 and returns top-k predicted labels."""
    pil_crop = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    tensor   = _transform(pil_crop).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits   = resnet_classify(tensor)
        topk_idx = torch.topk(logits, k=k, dim=1).indices[0].cpu().tolist()
    return [imagenet_categories[i].lower() for i in topk_idx]


def verify_species_top3(crop_bgr):
    """
    Returns (is_tiger: bool, top3_labels: list[str]).
    True if ResNet50 confirms a tiger is present.

    THREE-ATTEMPT STRATEGY with leopard/other-big-cat rejection:
      Attempt 1: top-3 on the preprocessed crop as-is.
      Attempt 2: if Attempt 1 fails AND crop is dark (< 60 brightness),
                 apply gamma + strong CLAHE, then check top-3 again.
      Attempt 3: if both fail, widen net to top-5 on the brightened crop.

    LEOPARD REJECTION RULE (applied before Attempt 3 widening):
      If top-3 are dominated by confirmed non-tiger big cats
      (leopard, snow leopard, jaguar, cheetah, cougar, lion),
      reject immediately — do NOT widen to top-5.

    TIGER CAT EXCLUSION RULE (Bug Fix 10):
      "tiger cat" (Leopardus tigrinus) is a small South American wild cat.
      "tiger shark" is a fish. "tiger beetle" is an insect.
      All contain the word "tiger" but are NOT tigers.
      The naive check `"tiger" in label` accepts all of these as tigers.
      Fix: _is_real_tiger_label() explicitly blocks known false-positive
      tiger-named species. Validated on e.jpeg (COCO_Proxy found a cat shape;
      species gate accepted "tiger cat" as confirmation → false positive).
    """
    # Labels that contain "tiger" but are NOT tigers — explicitly excluded
    TIGER_FALSE_POSITIVES = {"tiger cat", "tiger shark", "tiger beetle"}

    def _is_real_tiger_label(lbl):
        """
        True only if the label refers to an actual tiger (genus Panthera tigris).
        Blocks: tiger cat, tiger shark, tiger beetle — all contain "tiger" but
        are completely different animals.
        """
        lbl = lbl.lower()
        if any(fp in lbl for fp in TIGER_FALSE_POSITIVES):
            return False
        return "tiger" in lbl

    # Non-tiger big cats that should cause immediate rejection if they dominate
    NON_TIGER_BIG_CATS = {"leopard", "snow leopard", "jaguar", "cheetah",
                          "cougar", "lion", "puma", "panther", "ocelot"}

    def is_dominated_by_other_big_cats(labels):
        """True if 2 or more of top-3 are confirmed non-tiger big cats."""
        hits = sum(1 for l in labels
                   if any(nbc in l for nbc in NON_TIGER_BIG_CATS))
        return hits >= 2

    try:
        # Attempt 1 — top-3 on preprocessed crop
        top3_labels = _run_resnet_topk(crop_bgr, k=3)

        # TOP-1 BIG CAT REJECTION — checked BEFORE accepting any tiger in top-3.
        # A lion is NOT a tiger. A leopard is NOT a tiger. They look completely
        # different. If the model's single most confident prediction is a
        # non-tiger big cat, the image is either that animal or too ambiguous
        # to trust — either way, do NOT enroll as a tiger.
        #
        # BUG that was here: the old code checked `any tiger in top-3` FIRST and
        # returned True immediately — so ['lion','cougar','tiger'] was accepted
        # as a tiger because tiger appeared at position 3, without ever checking
        # that lion was the #1 prediction. Fixed: top-1 big cat rejection runs
        # BEFORE the tiger acceptance check.
        top1 = top3_labels[0].lower() if top3_labels else ""
        if any(nbc in top1 for nbc in NON_TIGER_BIG_CATS):
            return False, top3_labels  # top-1 is lion/leopard/jaguar etc. → reject

        if any(_is_real_tiger_label(lbl) for lbl in top3_labels):
            return True, top3_labels

        # Leopard rejection: if top-3 overwhelmingly says other big cat → reject
        # Do this BEFORE brightening — no amount of processing changes a leopard to a tiger
        if is_dominated_by_other_big_cats(top3_labels):
            return False, top3_labels  # e.g. [snow leopard, leopard, cougar] → false

        # Attempt 2 — brighten and retry top-3 for dark crops
        mean_bright = float(np.mean(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)))
        brightened  = enhance_dark_crop(crop_bgr) if mean_bright < 60 else crop_bgr
        if mean_bright < 60:
            top3_bright = _run_resnet_topk(brightened, k=3)
            if any(_is_real_tiger_label(lbl) for lbl in top3_bright):
                return True, [f"(brightened) {l}" for l in top3_bright]
            # Re-check leopard dominance on brightened crop too
            if is_dominated_by_other_big_cats(top3_bright):
                return False, top3_bright

        # Attempt 3 — widen to top-5 on brightened crop (catches unusual poses)
        # e.g. tiger lying on back → 'hog' at #1 but real tiger at #4
        # ONLY safe here because leopard-dominated images already rejected above
        top5_bright = _run_resnet_topk(brightened, k=5)
        if any(_is_real_tiger_label(lbl) for lbl in top5_bright):
            return True, [f"(top5) {l}" for l in top5_bright[:3]]

        # All three attempts failed — return Attempt 1 labels for debugging
        return False, top3_labels

    except Exception as e:
        return False, [f"error: {e}"]


# ══════════════════════════════════════════════════════════════════════════════
# FENCE DETECTION — Ground Truth Fix (000341.jpg)
# ──────────────────────────────────────────────────────────────────────────────
# Problem: A tiger standing behind a wire/bar fence has its body broken into
# segments by the fence bars. YOLO sees fragmented stripes that don't resemble
# a complete tiger body → misses the detection entirely.
# The uncovered-strip species gate also fails because the strip crop contains
# alternating bars and fur that ResNet50 cannot cleanly classify as "tiger".
#
# Fix: detect fence pattern via Canny edges + Hough vertical lines.
# If a fence is detected, apply morphological horizontal closing to "fill"
# the fence bars in the strip crop, then retry the species gate.
# Only applied to strips that already FAILED the species gate — so it cannot
# create false positives in normal (fence-free) images.
# ══════════════════════════════════════════════════════════════════════════════

def detect_fence_pattern(img_bgr, min_lines=6, min_line_len_ratio=0.25):
    """
    Returns True if a fence-like vertical line pattern is detected in img_bgr.

    Method:
      1. Convert to grayscale and apply Canny edge detection.
      2. Run Hough line transform to find long vertical lines.
      3. If >= min_lines vertical lines found AND they span >= min_line_len_ratio
         of image height → fence pattern confirmed.

    Why vertical lines: wire mesh fences, iron bar fences, and wooden slat fences
    all produce strong vertical edge signals when a tiger is photographed through them.

    Calibrated conservatively (min_lines=6) to avoid false triggers on tree trunks
    or single vertical structures that are not fences.
    """
    try:
        h, w = img_bgr.shape[:2]
        gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, threshold1=50, threshold2=150, apertureSize=3)

        # Hough lines — detect line segments
        min_len = int(h * min_line_len_ratio)   # line must span at least 25% of height
        lines   = cv2.HoughLinesP(edges,
                                   rho=1, theta=np.pi / 180,
                                   threshold=40,
                                   minLineLength=min_len,
                                   maxLineGap=10)
        if lines is None:
            return False

        # Count lines that are nearly vertical (angle within 15° of vertical)
        vertical_count = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            if dy > 0 and dx / dy < 0.27:   # tan(15°) ≈ 0.27
                vertical_count += 1

        return vertical_count >= min_lines

    except Exception:
        return False


def remove_fence_bars(crop_bgr):
    """
    Reduces fence bar influence in a crop by applying horizontal morphological
    closing — fills narrow vertical gaps (fence bars) with adjacent pixel values.

    Steps:
      1. Convert to LAB colour space for luminance processing.
      2. Apply horizontal closing with a 9×1 kernel (fills horizontal gaps).
      3. Apply gentle Gaussian blur to smooth transitions.
      4. Merge back to BGR.

    Why this works: fence bars create dark vertical stripes. Horizontal closing
    connects the tiger fur on both sides of a bar, producing a more complete
    tiger appearance that ResNet50 can classify more confidently.

    This is NOT applied to the original image — only to uncovered strip crops
    that already failed the species gate in a fence-detected image.
    """
    try:
        lab   = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        # Horizontal closing kernel: 1 row × 9 cols — fills narrow vertical bars
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
        l_closed = cv2.morphologyEx(l, cv2.MORPH_CLOSE, kernel)
        # Smooth to reduce artefacts from the closing operation
        l_smooth = cv2.GaussianBlur(l_closed, (3, 3), 0)
        merged   = cv2.merge([l_smooth, a, b])
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    except Exception:
        return crop_bgr   # on any failure, return original crop unchanged


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — VIEWPOINT CLASSIFICATION
# From Dr. Bose's architecture — separates Left_Flank / Right_Flank / Frontal.
#
# WHY THIS IS NEEDED:
# A tiger's left stripe pattern and right stripe pattern are asymmetric
# (like human fingerprints — left hand ≠ right hand). If TRACE sees Tiger_A
# from the left in image 1 and from the right in image 5, the cosine similarity
# between these two embeddings may be < 0.83. Result: TRACE creates Tiger_001
# and Tiger_002 for the same animal. With viewpoint-aware census, we know
# the True count = max(unique L IDs, unique R IDs) not L+R.
# ══════════════════════════════════════════════════════════════════════════════

def classify_viewpoint(x1, y1, x2, y2, img_w, img_h):
    """
    Classifies the viewing angle of a detected tiger crop.

    Logic:
      1. Aspect ratio: tall crop (height > 1.4x width) → Frontal
         A tiger standing upright and facing camera fills height > width.
      2. Horizontal position of crop centre:
         Left 40% of frame  → Left_Flank   (tiger body on left side of shot)
         Right 40% of frame → Right_Flank  (tiger body on right side)
         Middle 20%         → Frontal_Or_Ambiguous
    """
    crop_w           = x2 - x1
    crop_h           = y2 - y1
    crop_center_x    = (x1 + x2) / 2
    relative_center  = crop_center_x / img_w if img_w > 0 else 0.5

    # Tall crops: tiger is standing upright facing camera → Frontal
    if crop_h > 0 and crop_w > 0 and (crop_h / crop_w) > 1.4:
        return "Frontal"

    if relative_center < 0.40:
        return "Left_Flank"
    elif relative_center > 0.60:
        return "Right_Flank"
    else:
        return "Frontal"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — TIGER COLOUR MORPH CLASSIFICATION
# HSV-based — no training data required (from TRACE)
# ══════════════════════════════════════════════════════════════════════════════

def classify_color_morph(crop_bgr):
    """
    Classifies tiger colour variant using HSV channel statistics.
    Snow_White: very bright, near-zero saturation, uniform (rare captive/zoo tiger)
    White:      bright, low saturation
    Black:      very dark (pseudomelanistic — black background stripe dominates)
    Golden:     yellow-orange hue, low-medium saturation
    Orange:     everything else (standard Sundarbans Bengal tiger)
    """
    h, s, v = cv2.split(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV))
    mh, ms, mv, sv = np.mean(h), np.mean(s), np.mean(v), np.std(v)

    if mv > 220 and ms < 20  and sv < 15:  return "Snow_White"
    if mv > 180 and ms < 50:               return "White"
    if mv < 60:                             return "Black_Pseudomelanistic"
    if 25 <= mh <= 40 and ms < 120:        return "Golden"
    return "Orange_Standard"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8b — WATER REFLECTION GUARD
#
# RULE (permanent — do NOT remove):
#   A tiger's reflection in water must NEVER be counted as a second tiger.
#
# WHY THIS IS NEEDED:
#   Camera traps near rivers, ponds, and waterholes capture both the real tiger
#   AND its inverted mirror image in the water surface. The reflection:
#     • Passes the ResNet50 species gate  (stripe patterns survive flipping)
#     • Has fingerprint similarity ~0.45–0.60 with the real tiger
#     • Falls BELOW the grid-scan same-tiger threshold (0.68)
#   → Without this guard the grid scan wrongly enrolls the reflection as a new
#     individual (e.g. 006000.jpeg was counted as Tiger_001 + Tiger_079).
#
# HOW DETECTION WORKS:
#   Step 1 — detect_water_presence():
#     Scans the image for water signals:
#       • Blue/teal HSV pixels  (hue 90–130, low saturation)
#       • Dark reflective surface  (very dark + low saturation)
#       • Bottom half significantly darker than top half  (water sits below tiger)
#
#   Step 2 — is_water_reflection():
#     Flips crop_a along the split axis and extracts its ResNet50 embedding.
#     Compares with crop_b's embedding. Also checks the reverse direction.
#     If the best flip-similarity >= REFLECTION_FLIP_THRESH → mirror images.
#
#   Decision: water present  AND  mirror similarity high → reflection confirmed
#             → suppress the split → count as 1 tiger.
# ══════════════════════════════════════════════════════════════════════════════

REFLECTION_FLIP_THRESH = 0.70  # absolute floor — flip_sim must be at least this (ignored when using relative check)
# NOTE: in practice we use a RELATIVE check: flip_sim > direct_sim.
# A water reflection, when flipped back, looks MORE like the original than the
# unflipped version does. Two genuinely different tigers do NOT gain similarity
# when one is flipped — their flip_sim ≈ direct_sim or lower.
# The absolute REFLECTION_FLIP_THRESH is kept as a minimum sanity guard.


def detect_water_presence(img_bgr):
    """
    Returns True if water is likely visible in the image.

    Three signals are checked:
      1. Blue/teal pixels  — hue 90–130 in HSV with some saturation/brightness.
         Covers daylight water (rivers, ponds, puddles).
      2. Dark reflective surface  — very dark pixels with near-zero saturation.
         Covers night-time / shaded water that reads almost black.
      3. Bottom half much darker than top  — water pools at the bottom of the
         frame; a large brightness drop between top and bottom halves is a
         reliable structural signal even when colour cues are weak.
    """
    hsv              = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Signal 1: blue/teal water pixels
    blue_mask  = (h_ch >= 90) & (h_ch <= 130) & (s_ch >= 20) & (v_ch >= 20)
    blue_ratio = float(blue_mask.sum()) / float(blue_mask.size)

    # Signal 2: dark reflective surface (near-black, unsaturated)
    dark_mask  = (v_ch < 60) & (s_ch < 40)
    dark_ratio = float(dark_mask.sum()) / float(dark_mask.size)

    # Signal 3: bottom half meaningfully darker than top half
    ih           = img_bgr.shape[0]
    top_v        = float(np.mean(v_ch[:ih // 2, :]))
    bottom_v     = float(np.mean(v_ch[ih // 2:, :]))
    bottom_darker = (top_v - bottom_v) > 30

    return blue_ratio > 0.05 or dark_ratio > 0.20 or bottom_darker


def is_water_reflection(crop_a, crop_b, flip_axis):
    """
    Check whether crop_a and crop_b are mirror images of each other
    (indicating that one is a water reflection of the other).

    flip_axis:
      1 = horizontal flip  → use for LEFT / RIGHT grid-scan splits
      0 = vertical flip    → use for TOP  / BOTTOM grid-scan splits

    Method:
      • Flip crop_a along flip_axis → extract ResNet50 embedding.
      • Compare with crop_b's embedding (cosine similarity).
      • Repeat in the other direction (flip crop_b, compare with crop_a).
      • Take the maximum of both similarities.
      • If best_sim >= REFLECTION_FLIP_THRESH → reflection confirmed.

    Returns: (is_reflection: bool, best_flip_sim: float)
    """
    try:
        from sklearn.metrics.pairwise import cosine_similarity as _cos

        flipped_a    = cv2.flip(crop_a, flip_axis)
        fp_flipped_a = extract_fingerprint(flipped_a)
        fp_b         = extract_fingerprint(crop_b)
        sim_a2b      = float(_cos(fp_flipped_a.reshape(1, -1),
                                   fp_b.reshape(1, -1))[0][0])

        flipped_b    = cv2.flip(crop_b, flip_axis)
        fp_flipped_b = extract_fingerprint(flipped_b)
        fp_a         = extract_fingerprint(crop_a)
        sim_b2a      = float(_cos(fp_flipped_b.reshape(1, -1),
                                   fp_a.reshape(1, -1))[0][0])

        best = max(sim_a2b, sim_b2a)
        return best >= REFLECTION_FLIP_THRESH, best
    except Exception:
        return False, 0.0


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — GRAD-CAM EXPLAINABILITY
# From TRACE — "A model you can't explain is a model you can't trust."
#              Prof. Mahesh Balan, IIM Lucknow
#
# Grad-CAM tells us WHERE in the crop ResNet50 is looking when it makes its
# decision. Red = strong attention. Green = weak attention.
# Correct result: heatmap on stripes / body.
# Wrong result: heatmap on background (means fingerprint is unreliable).
# ══════════════════════════════════════════════════════════════════════════════

_fmaps, _grads = {}, {}

def _hook_fmaps(m, i, o):  _fmaps["l4"] = o
def _hook_grads(m, gi, go): _grads["l4"] = go[0]

# ResNet50: register Grad-CAM hooks on layer4 (deepest residual block)
resnet_classify.layer4.register_forward_hook(_hook_fmaps)
resnet_classify.layer4.register_full_backward_hook(_hook_grads)


def generate_gradcam(crop_bgr, save_path, label=""):
    """
    Grad-CAM on ResNet50 layer4 (deepest residual block).
    Saves side-by-side: Original | Heatmap | Overlay.
    """
    rgb    = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    tensor = _transform(Image.fromarray(rgb)).unsqueeze(0).to(DEVICE)
    tensor.requires_grad_(True)

    resnet_classify.zero_grad()
    out = resnet_classify(tensor)
    out[0, out.argmax(dim=1).item()].backward()

    weights = _grads["l4"].mean(dim=[2, 3], keepdim=True)
    cam     = torch.relu((weights * _fmaps["l4"]).sum(dim=1).squeeze())
    cam     = cam.detach().cpu().numpy()
    if cam.max() > 0:
        cam /= cam.max()

    h, w    = crop_bgr.shape[:2]
    heatmap = cv2.applyColorMap(
                cv2.resize((cam * 255).astype(np.uint8), (w, h)),
                cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(crop_bgr, 0.5, heatmap, 0.5, 0)

    if label:
        cv2.putText(overlay, label, (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    header = np.zeros((28, w * 3, 3), dtype=np.uint8)
    for i, txt in enumerate(["Original", "Grad-CAM", "Overlay"]):
        cv2.putText(header, txt, (i * w + w // 4, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    cv2.imwrite(str(save_path),
                np.vstack([header, np.hstack([crop_bgr, heatmap, overlay])]))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — ARTEFACT GUARDS
# Shadow, corner trace, overlap — all from TRACE (Section 7.5)
# ══════════════════════════════════════════════════════════════════════════════

def is_shadow_crop(crop_bgr):
    """
    Real tiger fur: high saturation (orange ≈ 120-180), medium brightness.
    Tiger shadow on ground: very low saturation AND very low brightness.
    This check rejects achromatic dark blobs that YOLO sometimes flags.
    """
    if crop_bgr.size == 0:
        return False
    hsv    = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    mean_s = float(np.mean(hsv[:, :, 1]))
    mean_v = float(np.mean(hsv[:, :, 2]))
    return mean_s < 30 and mean_v < 85


def is_corner_trace(x1, y1, x2, y2, img_w, img_h, margin=12):
    """
    Bounding box touching the image border within 12px = tiger entering/leaving.
    These crops are labelled Corner_Trace and never update the DB — only match.
    """
    return (x1 <= margin or y1 <= margin or
            x2 >= img_w - margin or y2 >= img_h - margin)


def check_box_overlap(accepted_boxes, new_box, iou_threshold=0.45):
    """
    Rejects new_box if it overlaps any already-accepted box above 45% IoU.
    Tiger shadows cast nearby have ~30-40% overlap — below YOLO's NMS but
    enough to flag as a duplicate via this post-hoc guard.
    """
    nx1, ny1, nx2, ny2 = new_box
    new_area = max(0, nx2 - nx1) * max(0, ny2 - ny1)
    if new_area == 0:
        return True

    for (ax1, ay1, ax2, ay2) in accepted_boxes:
        inter = max(0, min(nx2, ax2) - max(nx1, ax1)) * \
                max(0, min(ny2, ay2) - max(ny1, ay1))
        if inter == 0:
            continue
        union = new_area + (ax2 - ax1) * (ay2 - ay1) - inter
        if union > 0 and inter / union > iou_threshold:
            return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — CASCADE DETECTION
# The core innovation of this fused pipeline.
#
# Three detection stages tried in order:
#   Stage 1: YOLOv8n-OIV7  → knows "Tiger" directly (from Dr. Bose)
#   Stage 2: YOLOv8n-COCO  → proxy classes (from TRACE)
#   Stage 3: Whole image   → fallback if both YOLOs fail (from Yeshvir Script B)
#
# Species verification (Yeshvir Script B):
#   OIV7 boxes: no extra check needed (model confirmed "Tiger")
#   COCO boxes: ResNet50 top-3 must contain "tiger" before fingerprint
#   Fallback:   ResNet50 top-3 must contain "tiger" (only gate we have)
#
# This cascade means:
#   - Real tigers in clean images   → OIV7 catches them (best quality)
#   - Tigers in partial occlusion   → COCO proxy may catch what OIV7 missed
#   - Tiny/distant tigers           → fallback gives last chance with verification
# ══════════════════════════════════════════════════════════════════════════════

def _boxes_overlap(b1, b2, iou_thresh=0.30):
    """
    Returns True if two boxes overlap by more than iou_thresh (IoU).
    Used to deduplicate boxes found on raw vs preprocessed images.
    b1, b2 = (x1, y1, x2, y2) tuples.
    IoU = intersection area / union area.
    """
    ix1 = max(b1[0], b2[0]);  iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]);  iy2 = min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return False
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter) > iou_thresh


def run_cascade_detection(img_proc, img_h, img_w, img_raw=None):
    """
    Returns list of dicts:
    [{ "x1","y1","x2","y2", "conf", "detection_source", "requires_verify" }]

    detection_source: "OIV7_Direct" | "COCO_Proxy" | "Whole_Image_Fallback"
    requires_verify: True = ResNet50 top-3 must confirm tiger before accepting

    img_raw (optional): the original unprocessed image.
    If provided, OIV7 and COCO are also run on img_raw and any additional
    boxes not already covered by a preprocessed detection are merged in.
    WHY: CLAHE + Dark Channel Prior preprocessing can suppress the visual
    features that YOLO uses for detection (e.g. the second tiger in 15.jpeg
    disappears from OIV7 after preprocessing but is detected at conf=0.72
    on the raw image). Running on both images and merging via IoU deduplication
    ensures no tiger is missed due to preprocessing artifacts.
    """
    boxes_out = []

    def _run_oiv7(img):
        """Run OIV7 on an image, return list of Tiger boxes."""
        found = []
        if oiv7_detector is None:
            return found
        try:
            res = oiv7_detector(img, conf=CONF_OIV7, iou=IOU_THRESHOLD, verbose=False)[0]
            for box in res.boxes:
                if "tiger" in res.names[int(box.cls[0])].lower():
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    found.append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "conf": float(box.conf[0]),
                        "detection_source": "OIV7_Direct",
                        "requires_verify": False,
                    })
        except Exception:
            pass
        return found

    def _run_coco(img):
        """Run COCO on an image, return list of proxy-class boxes."""
        found = []
        try:
            res = coco_detector(img, conf=CONF_COCO, iou=IOU_THRESHOLD, verbose=False)[0]
            for box in res.boxes:
                if int(box.cls[0]) in TIGER_PROXY_CLASSES:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    found.append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "conf": float(box.conf[0]),
                        "detection_source": "COCO_Proxy",
                        "requires_verify": True,
                    })
        except Exception:
            pass
        return found

    def _merge_boxes(base_list, extra_list):
        """
        Add boxes from extra_list to base_list only if they don't overlap
        with any existing box (IoU < 0.30). This prevents double-counting
        when the same tiger is detected in both raw and preprocessed images.
        """
        for cand in extra_list:
            cand_tuple = (cand["x1"], cand["y1"], cand["x2"], cand["y2"])
            already_covered = any(
                _boxes_overlap(cand_tuple,
                               (b["x1"], b["y1"], b["x2"], b["y2"]))
                for b in base_list
            )
            if not already_covered:
                base_list.append(cand)
        return base_list

    # ── Stage 1: OIV7 — direct Tiger class ────────────────────────────────
    # Run on preprocessed image first (better contrast / dehazing).
    # Then also run on the raw image if provided — preprocessing can suppress
    # YOLO feature maps for distant or partially-occluded tigers.
    oiv7_proc_boxes = _run_oiv7(img_proc)
    boxes_out.extend(oiv7_proc_boxes)

    if img_raw is not None:
        oiv7_raw_boxes = _run_oiv7(img_raw)
        # Only add raw detections that are genuinely new (not already found)
        boxes_out = _merge_boxes(boxes_out, oiv7_raw_boxes)

    # ── Stage 2: COCO proxy — if OIV7 found nothing ───────────────────────
    # Run COCO on both images, same dual-scan logic as OIV7.
    if not boxes_out:
        coco_proc_boxes = _run_coco(img_proc)
        boxes_out.extend(coco_proc_boxes)

        if img_raw is not None and not boxes_out:
            # Only fall back to raw COCO if preprocessed COCO also found nothing.
            # For 16.jpeg: COCO finds 3 distant zebra-pattern boxes on the raw
            # image but preprocessing suppresses them. MIN_BOX_PARTIAL handles
            # the size filter downstream.
            coco_raw_boxes = _run_coco(img_raw)
            boxes_out = _merge_boxes(boxes_out, coco_raw_boxes)

    # ── Stage 3: Whole-image fallback — if both YOLOs found nothing ───────
    if not boxes_out:
        boxes_out.append({
            "x1": 0, "y1": 0, "x2": img_w, "y2": img_h,
            "conf": 0.0,
            "detection_source": "Whole_Image_Fallback",
            "requires_verify": True,  # ResNet top-3 is our only species gate
        })

    return boxes_out


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — SINGLE IMAGE ANALYSIS
# All stages combined into one function.
# ══════════════════════════════════════════════════════════════════════════════

def analyze_image(img_path, show_summary=True, run_gradcam=False):
    """
    Full pipeline for one image:
      Triage → Preprocess → Cascade Detect → Guards → Verify → Fingerprint →
      ViewPoint → Identity Match → Color Morph → Grad-CAM → Annotate
    Returns dict with all detections.
    """
    img_path = Path(img_path)
    result   = {
        "filename":       img_path.name,
        "tiger_count":    0,
        "detections":     [],
        "quality_bucket": "Unknown",
        "quality_issues": [],
        "skipped":        False,
    }

    # ── Load image ─────────────────────────────────────────────────────────
    img = cv2.imread(str(img_path))
    if img is None:
        try:
            img = cv2.cvtColor(np.array(Image.open(img_path).convert("RGB")),
                               cv2.COLOR_RGB2BGR)
        except Exception:
            return result

    # ── Phase 1: Triage ───────────────────────────────────────────────────
    bucket, reason, skip = triage_image(img)
    result["quality_bucket"] = bucket
    if skip:
        result["quality_issues"] = [reason]
        result["skipped"]        = True
        if show_summary:
            print(f"  SKIP  {img_path.name} — {bucket}: {reason}")
        return result

    # ── Phase 2: Preprocess ───────────────────────────────────────────────
    is_ir    = (bucket == "IR_Night")
    img_proc = preprocess(img, is_ir=is_ir)

    ih, iw   = img_proc.shape[:2]
    img_area = ih * iw
    annotated = img_proc.copy()

    # ── Phase 3: Cascade detection ────────────────────────────────────────
    # Pass the original unprocessed image (img) as img_raw so OIV7/COCO can
    # also scan the raw pixels. Some detections are suppressed by CLAHE + DCP
    # preprocessing — the dual-scan + IoU merge recovers them without
    # double-counting tigers that appear in both scans.
    raw_boxes = run_cascade_detection(img_proc, ih, iw, img_raw=img)

    if show_summary:
        src = raw_boxes[0]["detection_source"] if raw_boxes else "None"
        print(f"\n  {img_path.name}  [{bucket}]  |  "
              f"Detection stage: {src}  |  "
              f"{len([b for b in raw_boxes if b['detection_source'] != 'Whole_Image_Fallback'])} "
              f"YOLO box(es)")

    valid_count = 0

    for raw_box in raw_boxes:
        x1, y1 = raw_box["x1"], raw_box["y1"]
        x2, y2 = raw_box["x2"], raw_box["y2"]
        det_conf = raw_box["conf"]
        det_src  = raw_box["detection_source"]

        # ── Size filter ───────────────────────────────────────────────────
        box_ratio = (x2 - x1) * (y2 - y1) / img_area
        is_fallback_type = det_src in ("Whole_Image_Fallback",) or det_src.startswith("Grid_Scan")
        if not is_fallback_type and box_ratio < MIN_BOX_PARTIAL:
            continue

        # ── Visibility tier ───────────────────────────────────────────────
        if is_fallback_type:
            visibility = "Full_Body"
        else:
            visibility = "Full_Body" if box_ratio >= MIN_BOX_FULL else "Partial_Body"

        # ── Padded crop ───────────────────────────────────────────────────
        crop = img_proc[max(0, y1 - CROP_PAD): min(ih, y2 + CROP_PAD),
                        max(0, x1 - CROP_PAD): min(iw, x2 + CROP_PAD)]
        if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
            continue

        # ── Shadow guard ──────────────────────────────────────────────────
        # Skip shadow guard for IR_Night images: IR cameras produce greyscale
        # (zero saturation) images by design — they are NOT shadows. Applying
        # the shadow guard would reject every valid IR night tiger image because
        # saturation=0 always triggers the achromatic check.
        if bucket != "IR_Night" and is_shadow_crop(crop):
            if show_summary:
                print(f"    [SHADOW] rejected — achromatic dark crop")
            continue

        # ── Corner trace ──────────────────────────────────────────────────
        # BUG FIX: Whole-image fallback uses box (0,0,W,H) which always
        # touches the border → was being wrongly labeled Corner_Trace.
        # Fallback intentionally uses the full image → treat as Full_Body.
        if not is_fallback_type and is_corner_trace(x1, y1, x2, y2, iw, ih):
            visibility = "Corner_Trace"

        # ── Overlap guard ─────────────────────────────────────────────────
        accepted_boxes = [d["bbox"] for d in result["detections"]]
        if check_box_overlap(accepted_boxes, (x1, y1, x2, y2)):
            if show_summary:
                print(f"    [OVERLAP] rejected — duplicate/shadow near accepted box")
            continue

        # ── Species verification (Yeshvir Script B gate) ─────────────────
        # Applied to COCO proxy and whole-image fallback only.
        # OIV7 already confirmed Tiger — skip the check to avoid double compute.
        top3_labels = ["(OIV7_confirmed)"]
        _ir_species_bypass = False
        if raw_box["requires_verify"]:
            verified, top3_labels = verify_species_top3(crop)
            if not verified:
                # ── FIX F: IR Night species gate soft-bypass ─────────────
                # (Bug Fix 2026-06-10) IR Night images are greyscale and look
                # nothing like ImageNet colour tiger photos. ResNet50 returns
                # "ram / retriever / ox" for valid IR tiger images. If YOLO
                # detected an animal shape in an IR image, trust the detection
                # but flag it LOW confidence for human verification.
                # Without this fix: IR tigers hard-rejected → count = 0.
                if is_ir and det_conf >= 0.15:
                    _ir_species_bypass = True
                    if show_summary:
                        print(f"    [SPECIES-IR] species gate failed ({top3_labels}) "
                              f"but IR Night + YOLO conf={det_conf:.2f} — "
                              f"ACCEPTING with LOW confidence flag")
                # ── Bug Fix 19: Fence + Whole_Image_Fallback species bypass ──
                # When YOLO found nothing (fallback) AND a fence is present in
                # the image, the tiger may be occluded by fence bars showing
                # only back/tail — ResNet50 sees "shopping cart" not "tiger".
                # Accept with LOW confidence for human review.
                # Observed: 002600.jpg — tiger crossing fence, back+tail only.
                elif (det_src == "Whole_Image_Fallback"
                      and detect_fence_pattern(img_proc)):
                    _ir_species_bypass = True  # reuse the bypass flag
                    if show_summary:
                        print(f"    [SPECIES-FENCE] species gate failed ({top3_labels}) "
                              f"but fence detected + whole-image fallback — "
                              f"ACCEPTING with LOW confidence flag for human review")
                else:
                    if show_summary:
                        print(f"    [SPECIES] rejected — top-3: {top3_labels}")
                    continue
            if show_summary and not _ir_species_bypass:
                print(f"    [SPECIES] confirmed tiger — top-3: {top3_labels}")

        # ── Grid scan for multi-tiger frames (Whole_Image_Fallback only) ──
        # PROBLEM: When 2+ tigers share a frame and YOLO finds no individual
        # boxes, whole-image fallback counts the entire frame as 1 tiger.
        # Example: mother + cub walking together → both enrolled as 1 identity.
        #
        # FIX: After whole-image fallback confirms tiger, split the frame into
        # left/right halves AND top/bottom halves and run species gate on each
        # half independently. If BOTH halves of a split confirm tiger AND their
        # ResNet fingerprints are sufficiently different, replace the single
        # whole-frame detection with 2 separate half-frame detections.
        #
        # OVER-COUNTING GUARD (Bug Fix 9b):
        # A single large tiger can straddle both halves of the split (head in
        # left half, hindquarters in right half). Both halves pass the species
        # gate because both contain tiger body parts. Without this guard, one
        # tiger gets enrolled twice.
        # Guard: extract ResNet fingerprints from both halves. If cosine
        # similarity >= GRID_SAME_TIGER_THRESH → same individual spanning the
        # frame → keep as 1 detection (no split). Only split when fingerprints
        # are sufficiently different (genuinely different animals).
        #
        # Threshold calibrated from observed scores:
        #   Single tiger spanning frame → half_sim ≈ 0.72 (head vs hindquarters)
        #   Mother + cub side-by-side  → half_sim ≈ 0.39 (adult vs juvenile body)
        #   Two adult tigers similar    → half_sim ≈ 0.60–0.70 (different individuals)
        # 0.68 sits between 0.72 (same animal) and 0.39 (different animals).
        # NOTE: threshold was 0.85 originally (too high — single tiger at 0.72 < 0.85
        # was not caught). Lowered to 0.68 after calibration on Test MT dataset.
        #
        # ADDITIONAL BUG FIX (grid_handled flag — Bug Fix 9c):
        # Even with threshold=0.68, a second bug remained: when left/right guard
        # fires (same tiger, sim=0.72 >= 0.68), execution fell through to the
        # top/bottom split. Top half (head/back) + bottom half (belly/legs) of
        # the same tiger also both pass species gate, but sim=0.44 < 0.68 →
        # top/bottom incorrectly split the same animal. Fix: set grid_handled=True
        # when left/right reaches ANY conclusion (same or different tiger), and
        # gate the top/bottom block behind `if not grid_handled`.
        GRID_SAME_TIGER_THRESH = 0.68

        if det_src == "Whole_Image_Fallback" and iw >= 40 and ih >= 40:
            mid_x = iw // 2
            mid_y = ih // 2

            # grid_handled: set to True when left/right split reaches a definitive
            # conclusion (either "same tiger — keep 1" or "2 tigers — split L/R").
            # When True, the top/bottom split is SKIPPED entirely.
            # WHY THIS FLAG IS CRITICAL (Bug Fix 9c):
            #   A single tiger spanning the frame passes both the left/right AND
            #   top/bottom species gates (both halves always contain tiger body parts).
            #   The left/right fingerprint guard correctly catches it (sim=0.72 >= 0.68).
            #   BUT without this flag, execution falls through to the top/bottom split,
            #   which sees sim=0.44 < 0.68 and incorrectly creates 2 detections.
            #   Rule: left/right result is definitive — never attempt top/bottom
            #   if left/right already examined BOTH halves.
            grid_handled = False

            # V2 UPGRADE — Orientation-aware grid split
            # V1 always split Left/Right → caused false 2-tiger detection on
            # 000084.jpg (portrait/vertical image with 1 tiger).
            # V2: Portrait image (height > width) → split Top/Bottom
            #     Landscape image (width >= height) → split Left/Right
            is_portrait = ih > iw
            if is_portrait:
                # Portrait: tiger likely spans top-to-bottom → split Top/Bottom
                half_a = img_proc[:mid_y, :]   # top half
                half_b = img_proc[mid_y:, :]   # bottom half
                split_label_a, split_label_b = "Grid_Scan_T", "Grid_Scan_B"
                flip_axis_val = 0              # vertical flip for top/bottom reflection check
                box_a = {"x1": 0, "y1": 0,    "x2": iw, "y2": mid_y}
                box_b = {"x1": 0, "y1": mid_y, "x2": iw, "y2": ih}
            else:
                # Landscape: tiger likely spans left-to-right → split Left/Right
                half_a = img_proc[:, :mid_x]   # left half
                half_b = img_proc[:, mid_x:]   # right half
                split_label_a, split_label_b = "Grid_Scan_L", "Grid_Scan_R"
                flip_axis_val = 1              # horizontal flip for left/right reflection check
                box_a = {"x1": 0,    "y1": 0, "x2": mid_x, "y2": ih}
                box_b = {"x1": mid_x, "y1": 0, "x2": iw,   "y2": ih}

            left_half, right_half = half_a, half_b   # keep variable names for rest of logic
            left_ok,  _ = verify_species_top3(left_half)
            right_ok, _ = verify_species_top3(right_half)

            if left_ok and right_ok:
                # Both halves pass species gate — now check fingerprint similarity.
                # If fingerprints are too similar → same tiger spanning both halves
                # → do NOT split (over-counting guard).
                try:
                    fp_left  = extract_fingerprint(
                        enhance_dark_crop(left_half)
                        if float(np.mean(cv2.cvtColor(left_half, cv2.COLOR_BGR2GRAY))) < 50
                        else left_half)
                    fp_right = extract_fingerprint(
                        enhance_dark_crop(right_half)
                        if float(np.mean(cv2.cvtColor(right_half, cv2.COLOR_BGR2GRAY))) < 50
                        else right_half)
                    from sklearn.metrics.pairwise import cosine_similarity as _cos_sim
                    half_sim = float(_cos_sim(
                        fp_left.reshape(1, -1), fp_right.reshape(1, -1))[0][0])
                except Exception:
                    half_sim = 1.0  # if fingerprint fails, assume same tiger (safe)

                if half_sim >= GRID_SAME_TIGER_THRESH:
                    # Fingerprints too similar → single tiger spanning both halves.
                    # Mark as handled — do NOT attempt top/bottom split.
                    # A single tiger makes BOTH top and bottom halves pass the species
                    # gate too (head/back in top, body/legs in bottom). Without this
                    # flag the top/bottom split would fire at sim≈0.44 and enroll
                    # the same tiger twice as Tiger_059 and Tiger_060.
                    grid_handled = True
                    if show_summary:
                        print(f"    [GRID] both halves confirm tiger BUT "
                              f"fingerprint similarity={half_sim:.2f} >= {GRID_SAME_TIGER_THRESH}"
                              f" → same tiger spanning frame, keeping 1 detection")
                else:
                    # Fingerprints sufficiently different → COULD be 2 tigers,
                    # but first rule out water reflection (Bug Fix 12).
                    # A tiger standing near water produces a mirror image that:
                    #   • passes the species gate (stripes survive flipping)
                    #   • scores ~0.45–0.60 fingerprint similarity — below 0.68
                    # → without this guard the reflection is enrolled as a new tiger.
                    water_present = detect_water_presence(img_proc)
                    is_refl, flip_sim = is_water_reflection(
                        left_half, right_half, flip_axis=flip_axis_val)  # V2: orientation-aware
                    # RELATIVE CHECK: for a true reflection, flipping one crop makes
                    # it look MORE similar to the other (flip_sim > half_sim).
                    # Two genuinely different tigers do NOT gain similarity when
                    # one is flipped — their flip_sim ≈ direct_sim or lower.
                    is_actual_reflection = water_present and (flip_sim > half_sim)

                    grid_handled = True
                    if is_actual_reflection:
                        # Reflection confirmed — suppress split, keep 1 tiger.
                        if show_summary:
                            print(f"    [GRID] WATER REFLECTION detected (L/R) — "
                                  f"water present + flip_sim={flip_sim:.2f} > direct_sim={half_sim:.2f}"
                                  f" → suppressing split, 1 tiger kept")
                    else:
                        # Genuinely 2 tigers side-by-side.
                        if show_summary:
                            print(f"    [GRID] 2 tigers detected via left/right split "
                                  f"(half_sim={half_sim:.2f} < {GRID_SAME_TIGER_THRESH}) — "
                                  f"replacing whole-frame with 2 half-frame detections")
                        raw_boxes.append({
                            **box_a,
                            "conf": 0.0, "detection_source": split_label_a,
                            "requires_verify": False,
                            "_grid_verified": True,
                            "_top3": ["(grid_half_a_confirmed)"],
                        })
                        raw_boxes.append({
                            **box_b,
                            "conf": 0.0, "detection_source": split_label_b,
                            "requires_verify": False,
                            "_grid_verified": True,
                            "_top3": ["(grid_right_confirmed)"],
                        })
                        continue  # Skip whole-frame box — replaced by 2 halves above

            # Try vertical split (top | bottom) ONLY if left/right did not already
            # reach a conclusion. This prevents a single tiger from being split twice.
            if not grid_handled:
                top_half    = img_proc[:mid_y, :]
                bottom_half = img_proc[mid_y:, :]
                top_ok,    _ = verify_species_top3(top_half)
                bottom_ok, _ = verify_species_top3(bottom_half)

                if top_ok and bottom_ok:
                    # Same over-counting guard for top/bottom split
                    try:
                        fp_top    = extract_fingerprint(
                            enhance_dark_crop(top_half)
                            if float(np.mean(cv2.cvtColor(top_half, cv2.COLOR_BGR2GRAY))) < 50
                            else top_half)
                        fp_bottom = extract_fingerprint(
                            enhance_dark_crop(bottom_half)
                            if float(np.mean(cv2.cvtColor(bottom_half, cv2.COLOR_BGR2GRAY))) < 50
                            else bottom_half)
                        half_sim_tb = float(_cos_sim(
                            fp_top.reshape(1, -1), fp_bottom.reshape(1, -1))[0][0])
                    except Exception:
                        half_sim_tb = 1.0

                    if half_sim_tb >= GRID_SAME_TIGER_THRESH:
                        if show_summary:
                            print(f"    [GRID] top/bottom both confirm tiger BUT "
                                  f"similarity={half_sim_tb:.2f} → same tiger, keeping 1 detection")
                    else:
                        # Same water-reflection guard for top/bottom split.
                        # Reflection appears in the lower half (water is below tiger).
                        water_present_tb = detect_water_presence(img_proc)
                        is_refl_tb, flip_sim_tb = is_water_reflection(
                            top_half, bottom_half, flip_axis=0)  # 0 = vertical flip
                        is_actual_reflection_tb = (
                            water_present_tb and (flip_sim_tb > half_sim_tb))

                        if is_actual_reflection_tb:
                            if show_summary:
                                print(f"    [GRID] WATER REFLECTION detected (T/B) — "
                                      f"water present + flip_sim={flip_sim_tb:.2f} > direct_sim={half_sim_tb:.2f}"
                                      f" → suppressing split, 1 tiger kept")
                        else:
                            if show_summary:
                                print(f"    [GRID] 2 tigers detected via top/bottom split "
                                      f"(half_sim={half_sim_tb:.2f}) — "
                                      f"replacing whole-frame with 2 half-frame detections")
                            raw_boxes.append({
                                "x1": 0, "y1": 0, "x2": iw, "y2": mid_y,
                                "conf": 0.0, "detection_source": "Grid_Scan_T",
                                "requires_verify": False,
                                "_grid_verified": True,
                                "_top3": ["(grid_top_confirmed)"],
                            })
                            raw_boxes.append({
                                "x1": 0, "y1": mid_y, "x2": iw, "y2": ih,
                                "conf": 0.0, "detection_source": "Grid_Scan_B",
                                "requires_verify": False,
                                "_grid_verified": True,
                                "_top3": ["(grid_bottom_confirmed)"],
                            })
                            continue  # Skip whole-frame box — replaced by 2 halves above

        # For Grid_Scan boxes: top3_labels already set; species already verified
        if det_src.startswith("Grid_Scan"):
            top3_labels = raw_box.get("_top3", ["(grid_confirmed)"])
            visibility  = "Full_Body"  # each half is treated as full body

        # ── ViewPoint classification (Dr. Bose) ───────────────────────────
        viewpoint = classify_viewpoint(x1, y1, x2, y2, iw, ih)

        # ── Feature extraction + identity matching ────────────────────────
        # For dark crops: use enhanced (brightened) version for fingerprinting.
        # This ensures the 2048-dim feature vector encodes stripe texture
        # rather than near-uniform darkness. Same brightening logic as species gate.
        try:
            crop_mean = float(np.mean(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)))
            fp_input  = enhance_dark_crop(crop) if crop_mean < 50 else crop
            fp = extract_fingerprint(fp_input)
        except Exception as e:
            if show_summary:
                print(f"    [ERROR] fingerprint failed: {e}")
            continue

        is_partial = visibility in ("Partial_Body", "Corner_Trace")
        tiger_id, match_score, is_new = match_or_enroll(fp, update_db=not is_partial)

        if visibility == "Corner_Trace":
            id_status = f"Corner {'New' if is_new else f'Match {match_score:.2f}'}"
        elif visibility == "Partial_Body":
            id_status = f"Partial {'New' if is_new else f'Match {match_score:.2f}'}"
        else:
            id_status = "New Enrollment" if is_new else f"Recaptured {match_score:.2f}"

        # ── Colour morph ──────────────────────────────────────────────────
        morph = classify_color_morph(crop)
        tinfo = TIGER_TYPES.get(morph, {"display": morph, "color": (0, 255, 0)})

        # ── Grad-CAM (optional) ───────────────────────────────────────────
        if run_gradcam:
            gcam_path = GRADCAM_DIR / f"gradcam_{img_path.stem}_box{valid_count+1}.jpg"
            generate_gradcam(crop, gcam_path, label=f"{tiger_id}|{morph}|{viewpoint}")

        # ── Annotate ──────────────────────────────────────────────────────
        color     = (255, 255, 0) if visibility == "Corner_Trace" else tinfo["color"]
        thickness = 2 if visibility == "Corner_Trace" else 3
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        lbl1 = f"{tiger_id}  {tinfo['display']}  [{viewpoint}]"
        lbl2 = f"Src:{det_src[:4]}  Det:{det_conf:.0%}  Match:{match_score:.2f}  {visibility}"
        (lw, lh), _ = cv2.getTextSize(lbl1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        bg_top = max(0, y1 - lh * 2 - 14)
        cv2.rectangle(annotated, (x1, bg_top), (x1 + max(lw + 8, 260), y1), color, -1)
        tc = (0, 0, 0) if morph in ("White", "Snow_White", "Golden") else (255, 255, 255)
        cv2.putText(annotated, lbl1, (x1 + 4, y1 - lh - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, tc, 2)
        cv2.putText(annotated, lbl2, (x1 + 4, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, tc, 1)

        valid_count += 1
        _det_entry = {
            "tiger_id"        : tiger_id,
            "morph"           : morph,
            "display"         : tinfo["display"],
            "visibility"      : visibility,
            "viewpoint"       : viewpoint,
            "detection_source": det_src,
            "det_conf"        : round(det_conf, 4),
            "match_score"     : round(match_score, 4),
            "is_new"          : is_new,
            "id_status"       : id_status,
            "top3_labels"     : " | ".join(top3_labels),
            "bbox"            : (x1, y1, x2, y2),
        }
        # ── FIX F cont'd: flag IR-bypassed detections as LOW confidence ──
        if _ir_species_bypass:
            _det_entry["confidence"] = "LOW"
            _det_entry["ir_bypass"] = True
            result["needs_confirmation"] = True
            result.setdefault("uncertain_detections", [])
            result["uncertain_detections"].append(
                (tiger_id, "IR Night image — species gate failed, detection "
                 f"based on YOLO shape only (top-3: {' | '.join(top3_labels)}). "
                 "Human verification required."))
        result["detections"].append(_det_entry)

        if show_summary:
            print(f"    {tiger_id:<12} {tinfo['display']:<22} "
                  f"{visibility:<14} {viewpoint:<18} "
                  f"match={match_score:.2f}  {id_status}  [{det_src}]")

    # ── SAME-IMAGE CROSS-CHECK (Bug Fix 16) ─────────────────────────────────
    # PROBLEM: YOLO sometimes produces 2 boxes on the SAME tiger — one covering
    # the front, the other covering the back. IoU can be < 0.45 so the overlap
    # guard doesn't catch it. The fingerprints from different body parts can
    # differ enough (sim ~0.55–0.65) that each box gets a different tiger_id.
    # Result: 1 tiger counted as 2.
    #
    # Observed in 000985.jpg (Test Data 10): single tiger walking, YOLO produced
    # 2 boxes (front=Tiger_001, back=Tiger_004 at match=0.60).
    #
    # FIX: After all YOLO boxes are processed, compare each pair of detections'
    # crops directly. If their fingerprint similarity > 0.50 (same animal,
    # different body part range), merge by keeping the higher-confidence one
    # and removing the other.
    SAME_IMAGE_MERGE_THRESH = 0.50
    if len(result["detections"]) >= 2:
        from sklearn.metrics.pairwise import cosine_similarity as _cos_sim
        _to_remove = set()
        _dets = result["detections"]
        for _i in range(len(_dets)):
            if _i in _to_remove:
                continue
            for _j in range(_i + 1, len(_dets)):
                if _j in _to_remove:
                    continue
                # Skip if they already share the same tiger_id (Fix E handles count)
                if _dets[_i]["tiger_id"] == _dets[_j]["tiger_id"]:
                    continue
                # Extract crops and compute direct similarity
                _bi = _dets[_i]["bbox"]
                _bj = _dets[_j]["bbox"]
                _ci = img_proc[max(0, _bi[1]-CROP_PAD):min(ih, _bi[3]+CROP_PAD),
                               max(0, _bi[0]-CROP_PAD):min(iw, _bi[2]+CROP_PAD)]
                _cj = img_proc[max(0, _bj[1]-CROP_PAD):min(ih, _bj[3]+CROP_PAD),
                               max(0, _bj[0]-CROP_PAD):min(iw, _bj[2]+CROP_PAD)]
                if _ci.size == 0 or _cj.size == 0:
                    continue
                if _ci.shape[0] < 20 or _ci.shape[1] < 20:
                    continue
                if _cj.shape[0] < 20 or _cj.shape[1] < 20:
                    continue
                try:
                    _fp_i = extract_fingerprint(_ci)
                    _fp_j = extract_fingerprint(_cj)
                    _cross_sim = float(_cos_sim(
                        _fp_i.reshape(1, -1), _fp_j.reshape(1, -1))[0][0])
                except Exception:
                    continue
                if _cross_sim >= SAME_IMAGE_MERGE_THRESH:
                    # Keep the one with higher det_conf, remove the other
                    _keep, _drop = (_i, _j) if _dets[_i]["det_conf"] >= _dets[_j]["det_conf"] else (_j, _i)
                    _to_remove.add(_drop)
                    if show_summary:
                        print(f"    [CROSS-CHECK] {_dets[_drop]['tiger_id']} merged into "
                              f"{_dets[_keep]['tiger_id']} — same tiger, different body part "
                              f"(cross_sim={_cross_sim:.2f} >= {SAME_IMAGE_MERGE_THRESH})")
        if _to_remove:
            result["detections"] = [d for idx, d in enumerate(_dets) if idx not in _to_remove]
            valid_count = len(result["detections"])

    # ── Vertical Tiger Guard ──────────────────────────────────────────────────
    # RULE: All tigers in a frame must be in a roughly horizontal body orientation
    # for the multi-tiger scan (Low-Conf top-up + Remainder scan + Margin scan)
    # to be trusted.
    #
    # WHY: A tiger standing/posed vertically in frame produces a tall bounding box
    # (bbox_h >> bbox_w). When the Remainder scan splits the image into L/R/T/B
    # halves, it can find the top of the tiger in one half and the bottom in another
    # and misclassify them as two separate animals — as seen in 000084.jpg.
    #
    # RULE: If ANY accepted detection has bbox_height > bbox_width * VERTICAL_RATIO,
    # the tiger is considered vertically oriented. Skip Low-Conf top-up, Remainder
    # scan, and Margin scan entirely for this image, and flag it for human review.
    VERTICAL_RATIO = 1.3   # bbox height > 1.3× width → vertical orientation

    vertical_tiger_detected = False
    for _vd in result["detections"]:
        _bx1, _by1, _bx2, _by2 = _vd["bbox"]
        _bw = max(1, _bx2 - _bx1)
        _bh = max(1, _by2 - _by1)
        if _bh > _bw * VERTICAL_RATIO:
            vertical_tiger_detected = True
            break

    if vertical_tiger_detected:
        if show_summary:
            print(f"\n    [VERTICAL GUARD] Tall bounding box detected "
                  f"(tiger oriented vertically in frame).")
            print(f"    [VERTICAL GUARD] Skipping multi-tiger scan — "
                  f"flagging for human review.")

        # ── FIX: Hard-reject new enrollments with very low match scores ────────
        # Ground truth: 000363.jpg — V2 enrolled Tiger_003 (match=0.44) as a new
        # tiger under vertical guard. The vertical guard already suppresses the
        # multi-tiger split scan. A new enrollment with score < 0.50 under
        # vertical guard is almost certainly a false positive caused by the
        # vertically-oriented pose confusing the fingerprint extractor.
        # Action: REMOVE the low-score new enrollment entirely from detections
        # rather than just flagging it. This converts "warn and enroll" to
        # "hard reject" — matching the confidence flag rule in our memory.
        # Threshold 0.50: well above the normal false-positive range (< 0.40)
        # but below the first-sighting range (0.50–0.83) for genuinely new tigers.
        # ── FIX C: Soft-flag instead of hard-reject (Bug Fix 2026-06-10) ────
        # Previous behavior: new enrollments with match < 0.50 under vertical
        # guard were DELETED entirely — silently removing uncertain detections.
        # New behavior: flag them for human confirmation instead.
        # Rationale: per Confidence Flag Rule, report best-guess and ask human,
        # never silently delete. Tiger_011 in 000082.jpg scored 0.53 — barely
        # survived. Anything below 0.50 would have vanished without a trace.
        VERT_NEW_FLAG_THRESH = 0.50
        for _vd in result["detections"]:
            if _vd.get("is_new", False) and _vd.get("match_score", 0.0) < VERT_NEW_FLAG_THRESH:
                _vd["confidence"] = "LOW"
                if show_summary:
                    print(f"    [VERTICAL GUARD] LOW-CONF FLAG — {_vd['tiger_id']} "
                          f"new enrollment match={_vd.get('match_score',0):.2f} "
                          f"< {VERT_NEW_FLAG_THRESH} under vertical guard → flagged for human review.")

        result["needs_confirmation"]   = True
        result["uncertain_detections"] = result.get("uncertain_detections", [])
        for _vd in result["detections"]:
            if "confidence" not in _vd:
                _vd["confidence"] = "LOW"
            result["uncertain_detections"].append(
                (_vd["tiger_id"],
                 "tiger bounding box is taller than wide — vertical body orientation "
                 "detected; multi-tiger split skipped. Human confirmation required."))
        if show_summary:
            unique_ids   = list(dict.fromkeys(d["tiger_id"] for d in result["detections"]))
            unique_count = len(unique_ids)
            print(f"\n    {'=' * 58}")
            print(f"    ⚠  UNCERTAIN — HUMAN CONFIRMATION NEEDED")
            print(f"    {'=' * 58}")
            print(f"    MY PREDICTION : {unique_count} unique tiger(s) in this image")
            print(f"    PREDICTED IDs : {', '.join(unique_ids)}")
            print(f"    REASON        : Tiger body orientation is vertical.")
            print(f"                   Cannot safely run multi-tiger split.")
            print(f"    PLEASE CONFIRM: Is this count correct?")
            print(f"    {'=' * 58}")

    # ── Low-Confidence OIV7 Top-Up ──────────────────────────────────────────────
    # Problem: when two tigers overlap in the frame, YOLO's NMS keeps only the
    # high-confidence box and discards the lower-confidence second tiger (e.g.
    # 004840: main tiger conf=0.90 blocks second tiger conf=0.08 via NMS).
    #
    # Fix: run OIV7 again at conf=0.05 (vs. normal 0.25) and iou=0.45.
    # Any new Tiger boxes NOT substantially covered by existing detections are
    # treated as candidate second tigers and put through the full species +
    # fingerprint pipeline before enrolling.
    #
    # Threshold: LOWCONF_COVERAGE_THRESH = 0.45
    #   If an existing accepted box covers >= 45% of the low-conf box area →
    #   assume it's the same animal already counted.  Below 45% → new candidate.
    LOWCONF_CONF_THRESH  = 0.05
    LOWCONF_IOU_NMS      = 0.45
    LOWCONF_SAME_BOX_IOU = 0.50   # IoU above which a low-conf box == existing box (skip)
    # WHY IoU (not coverage): a small second-tiger box in the corner of a large
    # main-tiger box has HIGH area-coverage (~88%) but LOW IoU (~0.18) — they're
    # clearly different detections. IoU correctly sees them as distinct.

    # ── FIX B: Allow LowConf scan even when vertical guard triggered ─────
    # (Bug Fix 2026-06-10) A vertical tiger doesn't mean there's no OTHER
    # tiger at low confidence in a DIFFERENT part of the frame. We now allow
    # the scan but require IoU < 0.30 with the vertical tiger's bbox to
    # ensure we're not re-detecting the same animal from a different angle.
    if valid_count >= 1 and oiv7_detector is not None:
        accepted_boxes = [d["bbox"] for d in result["detections"]]

        def _max_iou_with_accepted(bx1, by1, bx2, by2):
            """Max IoU between candidate box and any accepted box."""
            box_area = max(1, (bx2 - bx1) * (by2 - by1))
            max_iou  = 0.0
            for (ax1, ay1, ax2, ay2) in accepted_boxes:
                ix1 = max(bx1, ax1); iy1 = max(by1, ay1)
                ix2 = min(bx2, ax2); iy2 = min(by2, ay2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                if inter == 0:
                    continue
                union = box_area + (ax2 - ax1) * (ay2 - ay1) - inter
                max_iou = max(max_iou, inter / max(1, union))
            return max_iou

        try:
            # Use raw image (not img_proc): CLAHE preprocessing can drop weak
            # second-tiger detections from conf=0.08 to undetectable.
            lc_res = oiv7_detector(img, verbose=False,
                                   conf=LOWCONF_CONF_THRESH,
                                   iou=LOWCONF_IOU_NMS)[0]
        except Exception as _lc_err:
            if show_summary:
                print(f"    [LOWCONF] OIV7 low-conf run failed: {_lc_err}")
            lc_res = None

        _TIGER_FP = {"tiger cat", "tiger shark", "tiger beetle"}

        def _is_tiger_lbl(lbl):
            lbl = lbl.lower()
            return "tiger" in lbl and not any(fp in lbl for fp in _TIGER_FP)

        if lc_res is not None and lc_res.boxes is not None:
            lc_tiger_boxes = []
            for lc_box, lc_cls, lc_conf in zip(
                    lc_res.boxes.xyxy.cpu().numpy(),
                    lc_res.boxes.cls.cpu().numpy(),
                    lc_res.boxes.conf.cpu().numpy()):
                lc_label = oiv7_detector.names[int(lc_cls)]
                if _is_tiger_lbl(lc_label):
                    lc_tiger_boxes.append((lc_box, float(lc_conf)))

            for (lc_box, lc_conf_val) in lc_tiger_boxes:
                lx1, ly1, lx2, ly2 = [int(v) for v in lc_box]
                lx1 = max(0, lx1 - CROP_PAD); ly1 = max(0, ly1 - CROP_PAD)
                lx2 = min(iw, lx2 + CROP_PAD); ly2 = min(ih, ly2 + CROP_PAD)
                if lx2 <= lx1 or ly2 <= ly1:
                    continue

                box_iou = _max_iou_with_accepted(lx1, ly1, lx2, ly2)
                # When vertical guard is active, use stricter IoU threshold (0.30)
                # to prevent re-detecting the same vertical tiger from a slightly
                # different crop angle. When NOT vertical, use normal threshold.
                _iou_thresh = 0.30 if vertical_tiger_detected else LOWCONF_SAME_BOX_IOU
                if box_iou >= _iou_thresh:
                    if show_summary:
                        print(f"    [LOWCONF] ({lx1},{ly1})-({lx2},{ly2}) "
                              f"conf={lc_conf_val:.2f}: same box as existing "
                              f"(IoU={box_iou:.2f}) — skipped")
                    continue

                lc_crop = img_proc[ly1:ly2, lx1:lx2]
                if lc_crop.shape[0] < 20 or lc_crop.shape[1] < 20:
                    continue

                # Species gate
                lc_ok, lc_top3 = verify_species_top3(lc_crop)
                if not lc_ok:
                    if show_summary:
                        print(f"    [LOWCONF] ({lx1},{ly1})-({lx2},{ly2}) "
                              f"conf={lc_conf_val:.2f}: species rejected — {lc_top3}")
                    continue

                # Fingerprint
                try:
                    mean_lc = float(np.mean(cv2.cvtColor(lc_crop, cv2.COLOR_BGR2GRAY)))
                    fp_lc = extract_fingerprint(
                        enhance_dark_crop(lc_crop) if mean_lc < 50 else lc_crop)
                except Exception:
                    continue

                # Similarity guard vs. existing detections
                from sklearn.metrics.pairwise import cosine_similarity as _cos_sim
                max_sim_lc = 0.0
                for existing_det in result["detections"]:
                    tid = existing_det.get("tiger_id", "")
                    if tid and tid in tiger_db:
                        fp_ex = tiger_db[tid]
                    else:
                        ex1, ey1, ex2, ey2 = existing_det["bbox"]
                        ex_crop = img_proc[max(0, ey1-CROP_PAD):min(ih, ey2+CROP_PAD),
                                           max(0, ex1-CROP_PAD):min(iw, ex2+CROP_PAD)]
                        if ex_crop.size == 0:
                            continue
                        try:
                            mean_ec = float(np.mean(
                                cv2.cvtColor(ex_crop, cv2.COLOR_BGR2GRAY)))
                            fp_ex = extract_fingerprint(
                                enhance_dark_crop(ex_crop) if mean_ec < 50 else ex_crop)
                        except Exception:
                            continue
                    try:
                        sim = float(_cos_sim(fp_lc.reshape(1, -1),
                                             fp_ex.reshape(1, -1))[0][0])
                        max_sim_lc = max(max_sim_lc, sim)
                    except Exception:
                        pass

                LOWCONF_SAME_TIGER_THRESH = 0.82
                if max_sim_lc >= LOWCONF_SAME_TIGER_THRESH:
                    if show_summary:
                        print(f"    [LOWCONF] ({lx1},{ly1})-({lx2},{ly2}) "
                              f"conf={lc_conf_val:.2f}: same tiger (sim={max_sim_lc:.2f}) "
                              f"— skipped")
                    continue

                if show_summary:
                    print(f"    [LOWCONF] ({lx1},{ly1})-({lx2},{ly2}) "
                          f"conf={lc_conf_val:.2f}: new tiger "
                          f"(IoU={box_iou:.2f}, max_sim={max_sim_lc:.2f})")

                if run_gradcam:
                    gcam_path = (GRADCAM_DIR /
                                 f"gradcam_{img_path.stem}_lowconf{valid_count+1}.jpg")
                    generate_gradcam(lc_crop, gcam_path,
                                     label=f"LowConf|{lc_conf_val:.2f}")

                try:
                    fp_enrol_lc = extract_fingerprint(
                        enhance_dark_crop(lc_crop) if mean_lc < 50 else lc_crop)
                    lc_id, lc_score, lc_new = match_or_enroll(fp_enrol_lc,
                                                               update_db=True)
                except Exception as e:
                    if show_summary:
                        print(f"    [LOWCONF] enroll failed: {e}")
                    continue

                lc_morph  = classify_color_morph(lc_crop)
                lc_tinfo  = TIGER_TYPES.get(lc_morph, {"display": lc_morph,
                                                        "color": (255, 165, 0)})
                lc_vp     = classify_viewpoint(lx1, ly1, lx2, ly2, iw, ih)
                lc_status = "New Enrollment" if lc_new else f"Recaptured {lc_score:.2f}"

                cv2.rectangle(annotated, (lx1, ly1), (lx2, ly2), lc_tinfo["color"], 3)
                llbl1 = f"{lc_id}  {lc_tinfo['display']}  [{lc_vp}]"
                llbl2 = f"Src:LowConf{lc_conf_val:.2f}  Match:{lc_score:.2f}  {lc_status}"
                (llw, llh), _ = cv2.getTextSize(llbl1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                lbg_top = max(0, ly1 - llh * 2 - 14)
                cv2.rectangle(annotated, (lx1, lbg_top),
                              (lx1 + max(llw + 8, 260), ly1), lc_tinfo["color"], -1)
                ltc = (0, 0, 0) if lc_morph in ("White", "Snow_White", "Golden") \
                               else (255, 255, 255)
                cv2.putText(annotated, llbl1, (lx1 + 4, ly1 - llh - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, ltc, 2)
                cv2.putText(annotated, llbl2, (lx1 + 4, ly1 - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, ltc, 1)

                accepted_boxes.append((lx1, ly1, lx2, ly2))
                valid_count += 1
                result["detections"].append({
                    "tiger_id"        : lc_id,
                    "morph"           : lc_morph,
                    "display"         : lc_tinfo["display"],
                    "visibility"      : "Partial_Body",
                    "viewpoint"       : lc_vp,
                    "detection_source": f"LowConf_OIV7_{lc_conf_val:.2f}",
                    "det_conf"        : round(lc_conf_val, 4),
                    "match_score"     : round(lc_score, 4),
                    "is_new"          : lc_new,
                    "id_status"       : lc_status,
                    "top3_labels"     : " | ".join(lc_top3),
                    "bbox"            : (lx1, ly1, lx2, ly2),
                })

                if show_summary:
                    print(f"    {lc_id:<12} {lc_tinfo['display']:<22} "
                          f"Partial_Body   {lc_vp:<18} "
                          f"match={lc_score:.2f}  {lc_status}  "
                          f"[LowConf_OIV7_{lc_conf_val:.2f}]")

    # ── Remainder scan — finds tigers YOLO missed in a multi-tiger frame ────
    # BUG (Fix 11): When YOLO draws a box on one tiger it stops. Grid scan only
    # fires for Whole_Image_Fallback (0 YOLO boxes). So frames like b.jpeg
    # (adult + cub) or c.jpeg (3 tigers at a kill) produce only 1 detection
    # because YOLO locked onto the dominant animal and the other halves were
    # never examined.
    #
    # FIX: After all accepted detections are finalised, split the image into
    # four halves (L / R / T / B) and check each half that is NOT already
    # substantially covered by an accepted bounding box. If species gate
    # confirms a tiger in an uncovered half AND its fingerprint is different
    # from every already-detected tiger in this image → new individual found.
    #
    # COVERAGE THRESHOLD (REMAINDER_COVERAGE_THRESH = 0.65):
    #   A half is considered "too covered" only if existing boxes overlap >= 65% of it.
    #   WHY 0.65 (raised from original 0.30):
    #     In close-up camera trap shots, the dominant tiger fills most of the frame.
    #     Its OIV7 box covers 40-60% of every half. With threshold=0.30, ALL halves
    #     were silently skipped — the remainder scan never ran, missing background/
    #     secondary tigers. Raising to 0.65 allows halves with 30-64% coverage to
    #     be examined. The SIMILARITY GUARD (below) is then responsible for blocking
    #     double-counts — it is the real safety net, not coverage.
    #     Validated on: 004840.jpg, 004880.jpg (foreground + background tiger pairs).
    #
    # SIMILARITY GUARD (REMAINDER_SAME_TIGER_THRESH = 0.84):
    #   Prevents double-counting the foreground tiger that spills into a covered half.
    #   WHY 0.84 (not the 0.68 used in grid scan):
    #     Grid scan compares two HALF-IMAGES against each other (symmetric).
    #     Remainder scan compares a tight YOLO CROP against a HALF-IMAGE that
    #     contains background + another tiger. The tight-crop vs half-image
    #     comparison always reads higher similarity because the half-image
    #     "dilutes" tiger features with background.
    #     Calibrated values:
    #       Foreground tiger crop vs half with background tiger → sim ≈ 0.79–0.83
    #       Same tiger spanning halves (single animal)          → sim ≈ 0.85–0.90
    #     0.84 sits between these two ranges.
    REMAINDER_COVERAGE_THRESH   = 0.65
    # Bug Fix 20: Lowered from 0.82 → 0.80. A lying tiger's hindquarters
    # scored sim=0.81 in 002627.jpg and slipped through at 0.82. Different
    # individuals score 0.35–0.70 so 0.80 still safely separates them.
    REMAINDER_SAME_TIGER_THRESH = 0.80   # guard vs. YOLO-detected tigers in this image
    REMAINDER_CROSS_HALF_THRESH = 0.55   # guard vs. OTHER remainder-enrolled tigers
    #   Remainder halves are full-image splits (include background), so similarity
    #   is lower than grid-crop halves. Calibration:
    #   Single tiger spanning frame → L/R cross-half sim ~0.65  (must block: > 0.55 ✓)
    #   Same cub top/bottom half cross-sim              → ~0.58  (must block: > 0.55 ✓)
    #   Adult tiger vs cub (different individuals) → ~0.35–0.45 (must pass: < 0.55 ✓)
    #   Lowered from 0.62 → 0.55 to catch same-cub T/B pairs (b.jpeg: sim=0.58)

    # ── FIX G: Allow Remainder scan even when vertical guard triggered ───
    # (Bug Fix 2026-06-10) A vertical tiger doesn't mean there are no OTHER
    # tigers elsewhere in the frame. We now allow the scan but:
    # 1) Use stricter coverage threshold (0.75) when vertical — vertical tiger
    #    occupies more vertical space, so halves with high coverage are more
    #    likely to be re-detecting the same animal.
    # 2) Use stricter similarity threshold (0.75) to prevent the vertical
    #    tiger's head/tail from being enrolled as separate animals.
    if valid_count >= 1 and iw >= 80 and ih >= 80:
        from sklearn.metrics.pairwise import cosine_similarity as _cos_sim

        accepted_coords = [d["bbox"] for d in result["detections"]]
        remainder_enrolled_fps = []   # fingerprints of tigers newly enrolled via remainder scan

        def _half_coverage(hx1, hy1, hx2, hy2):
            """Fraction of this half area already covered by accepted boxes."""
            half_area = max(1, (hx2 - hx1) * (hy2 - hy1))
            covered = 0
            for (ax1, ay1, ax2, ay2) in accepted_coords:
                ix1 = max(hx1, ax1); iy1 = max(hy1, ay1)
                ix2 = min(hx2, ax2); iy2 = min(hy2, ay2)
                covered += max(0, ix2 - ix1) * max(0, iy2 - iy1)
            return covered / half_area

        mid_x = iw // 2
        mid_y = ih // 2

        remainder_halves = [
            (0,     0,     mid_x, ih,    "Remainder_L", img_proc[:, :mid_x]),
            (mid_x, 0,     iw,    ih,    "Remainder_R", img_proc[:, mid_x:]),
            (0,     0,     iw,    mid_y, "Remainder_T", img_proc[:mid_y, :]),
            (0,     mid_y, iw,    ih,    "Remainder_B", img_proc[mid_y:, :]),
        ]

        # FIX G cont'd: stricter thresholds under vertical guard
        _rem_cov_thresh = 0.75 if vertical_tiger_detected else REMAINDER_COVERAGE_THRESH
        _rem_sim_thresh = 0.75 if vertical_tiger_detected else REMAINDER_SAME_TIGER_THRESH

        for (rx1, ry1, rx2, ry2, rsrc, half_img) in remainder_halves:
            # Skip halves already substantially covered by an accepted detection
            cov = _half_coverage(rx1, ry1, rx2, ry2)
            if cov >= _rem_cov_thresh:
                if show_summary:
                    print(f"    [REMAINDER] {rsrc}: skipped — {cov:.0%} covered "
                          f"(threshold {REMAINDER_COVERAGE_THRESH:.0%})")
                continue
            if half_img.shape[0] < 20 or half_img.shape[1] < 20:
                continue

            # ── CORNER / PARTIAL OVERLAP GUARD ───────────────────────────────
            # Problem: a Corner_Trace or Partial_Body tiger (e.g. animal entering
            # the frame from the left) is detected by YOLO with a small bbox that
            # clips the edge. The adjacent half-image contains the REST of the same
            # tiger's body. The fingerprint similarity between a tiny corner crop and
            # a full-half image is only ~0.55–0.65 — below the 0.82 same-tiger
            # threshold — so the Remainder scan wrongly enrolls it as a new individual.
            #
            # Fix: if the remainder half OVERLAPS an existing Corner_Trace or
            # Partial_Body detection by >= 25% of THAT detection's area, the half
            # is the continuation of the same animal's body — skip it.
            #
            # 25% threshold calibration:
            #   000466 Corner_Trace overlap with Remainder_R → 30% → BLOCKED ✓
            #   Genuine second tiger overlap with existing small box → < 10% → PASSES ✓
            CORNER_OVERLAP_BLOCK_THRESH = 0.25
            _corner_block = False
            for _cd in result["detections"]:
                if _cd.get("visibility") not in ("Corner_Trace", "Partial_Body"):
                    continue
                cx1, cy1, cx2, cy2 = _cd["bbox"]
                det_area = max(1, (cx2 - cx1) * (cy2 - cy1))
                ox1 = max(rx1, cx1); oy1 = max(ry1, cy1)
                ox2 = min(rx2, cx2); oy2 = min(ry2, cy2)
                overlap = max(0, ox2 - ox1) * max(0, oy2 - oy1)
                if overlap / det_area >= CORNER_OVERLAP_BLOCK_THRESH:
                    _corner_block = True
                    if show_summary:
                        print(f"    [REMAINDER] {rsrc}: CORNER OVERLAP guard — "
                              f"half contains {overlap/det_area:.0%} of "
                              f"{_cd['tiger_id']} Corner/Partial bbox → same animal, skipped")
                    break
            if _corner_block:
                continue

            # Species gate on uncovered half
            rem_ok, rem_top3 = verify_species_top3(half_img)
            if not rem_ok:
                if show_summary:
                    print(f"    [REMAINDER] {rsrc} (cov={cov:.0%}): "
                          f"species rejected — {rem_top3}")
                continue

            # ── Uncovered sub-region gate ─────────────────────────────────────
            # The full-half gate often passes because the MAIN TIGER occupies
            # 40-64% of the half and its body texture drives the "tiger" label.
            # A second species gate on ONLY the uncovered portion (the pixels NOT
            # covered by any accepted box) eliminates this bias.
            #
            # Algorithm: find the largest rectangular strip within this half that
            # has no overlap with any accepted box, then re-verify species on it.
            # If that strip is too small or doesn't confirm "tiger" → false positive.
            #
            # Coverage guard: only apply strip gate when cov >= 40%.
            # When coverage is low (< 40%) the main tiger occupies only a small
            # fraction of the half — a cub or partly-visible second tiger may
            # genuinely be in the remainder but too small to survive strip cropping.
            # For those halves, trust the full-half species gate result above.
            # Known calibration:
            #   b.jpeg cub:     cov = 22-27%  → gate skipped → cub found ✓
            #   false positives: cov = 47-64% → gate applied → false pos caught ✓
            UNCOV_MIN_DIM        = 50    # px; strip must be ≥ 50px in both dimensions
            UNCOV_APPLY_COV_THRESH = 0.28  # only run strip gate if cov ≥ 28%
            # Calibration: b.jpeg cub halves cov=22-26% → gate skipped → cub enrolls ✓
            #              000006 false pos  cov=30%    → gate applied → false pos caught ✓
            #              000036/000158 FPs cov=47-64% → gate applied → caught ✓

            _overlap_boxes = [(ax1, ay1, ax2, ay2)
                              for (ax1, ay1, ax2, ay2) in accepted_coords
                              if ax1 < rx2 and ax2 > rx1
                              and ay1 < ry2 and ay2 > ry1]

            if _overlap_boxes and cov >= UNCOV_APPLY_COV_THRESH:
                # Union bounding box of overlapping accepted detections within this half
                _ux1 = max(rx1, min(b[0] for b in _overlap_boxes))
                _uy1 = max(ry1, min(b[1] for b in _overlap_boxes))
                _ux2 = min(rx2, max(b[2] for b in _overlap_boxes))
                _uy2 = min(ry2, max(b[3] for b in _overlap_boxes))

                # Four candidate strips outside the union box, clipped to this half
                _cand_strips = []
                if _ux1 - rx1 >= UNCOV_MIN_DIM:
                    _cand_strips.append((rx1, ry1, _ux1, ry2))           # left strip
                if rx2 - _ux2 >= UNCOV_MIN_DIM:
                    _cand_strips.append((_ux2, ry1, rx2, ry2))           # right strip
                if _uy1 - ry1 >= UNCOV_MIN_DIM:
                    _cand_strips.append((rx1, ry1, rx2, _uy1))           # top strip
                if ry2 - _uy2 >= UNCOV_MIN_DIM:
                    _cand_strips.append((rx1, _uy2, rx2, ry2))           # bottom strip

                if not _cand_strips:
                    # No uncovered strip is large enough — saturated with existing tiger
                    if show_summary:
                        print(f"    [REMAINDER] {rsrc} (cov={cov:.0%}): "
                              f"no viable uncovered sub-region — skipped")
                    continue

                # Check ALL viable strips — a second tiger may be in any one of them.
                # Only reject if NONE of the strips confirms "tiger".
                # (Largest-strip-only approach missed the cub in b.jpeg which was in
                #  a smaller strip while the largest strip was empty background.)
                _any_strip_confirmed = False
                for (_bx1, _by1, _bx2, _by2) in _cand_strips:
                    if (_bx2 - _bx1) < UNCOV_MIN_DIM or (_by2 - _by1) < UNCOV_MIN_DIM:
                        continue
                    _ucrop = img_proc[_by1:_by2, _bx1:_bx2]
                    _ucrop_ok, _ucrop_top3 = verify_species_top3(_ucrop)
                    if _ucrop_ok:
                        _any_strip_confirmed = True
                        if show_summary:
                            print(f"    [REMAINDER] {rsrc} (cov={cov:.0%}): "
                                  f"uncovered sub-region confirmed — {_ucrop_top3}")
                        break  # one confirmed strip is enough

                if not _any_strip_confirmed:
                    # ── FENCE-AWARE RETRY (Ground Truth Fix — 000341.jpg) ──────
                    # Problem: a tiger behind a wire/bar fence has its body broken
                    # into fragments by the fence bars. The uncovered strip crop
                    # shows alternating bars + fur — ResNet50 cannot classify this
                    # as "tiger" because the texture is corrupted by the bars.
                    #
                    # Two-stage fix:
                    #   Stage 1 — ResNet50 species gate on fence-cleaned strip.
                    #   Stage 2 — If ResNet50 still fails, run YOLO at conf=0.01
                    #             on the cleaned strip. WHY: fence bars fragment the
                    #             tiger body into disconnected segments. YOLO at
                    #             normal conf=0.05 merges these segments into one low-
                    #             confidence box which NMS suppresses.  At conf=0.01
                    #             on the bar-removed crop, YOLO can recover the box.
                    #
                    # Safety: both stages ONLY run when:
                    #   (a) all strips already failed the normal gate, and
                    #   (b) a fence pattern is confirmed by Hough line detection.
                    _fence_confirmed = False
                    if detect_fence_pattern(img_proc):
                        if show_summary:
                            print(f"    [FENCE] Fence pattern detected in image — "
                                  f"retrying with fence-bar reduction (ResNet50 + YOLO@0.01).")
                        _FENCE_TIGER_FP = {"tiger cat", "tiger shark", "tiger beetle"}
                        for (_bx1, _by1, _bx2, _by2) in _cand_strips:
                            if (_bx2 - _bx1) < UNCOV_MIN_DIM or (_by2 - _by1) < UNCOV_MIN_DIM:
                                continue
                            _raw_strip   = img_proc[_by1:_by2, _bx1:_bx2]
                            _clean_strip = remove_fence_bars(_raw_strip)

                            # Stage 1: ResNet50 species gate on cleaned strip
                            _fence_ok, _fence_top3 = verify_species_top3(_clean_strip)
                            if _fence_ok:
                                _fence_confirmed    = True
                                _any_strip_confirmed = True
                                if show_summary:
                                    print(f"    [FENCE] ResNet50 confirmed tiger in "
                                          f"cleaned strip — {_fence_top3}")
                                break

                            # Stage 2: YOLO ultra-low-conf (conf=0.01) on cleaned strip
                            # Triggered only when ResNet50 can't see the fence-occluded body
                            if oiv7_detector is not None:
                                try:
                                    _fc_res = oiv7_detector(
                                        _clean_strip, verbose=False,
                                        conf=0.01, iou=0.45)[0]
                                    if _fc_res.boxes is not None:
                                        for _fc_cls, _fc_conf in zip(
                                                _fc_res.boxes.cls.cpu().numpy(),
                                                _fc_res.boxes.conf.cpu().numpy()):
                                            _fc_lbl = oiv7_detector.names[int(_fc_cls)]
                                            _fc_lbl_l = _fc_lbl.lower()
                                            if ("tiger" in _fc_lbl_l and
                                                    not any(fp in _fc_lbl_l
                                                            for fp in _FENCE_TIGER_FP)):
                                                _fence_confirmed    = True
                                                _any_strip_confirmed = True
                                                if show_summary:
                                                    print(f"    [FENCE] YOLO@0.01 confirmed "
                                                          f"tiger in cleaned strip "
                                                          f"(conf={_fc_conf:.3f}) — {_fc_lbl}")
                                                break
                                except Exception as _fc_err:
                                    if show_summary:
                                        print(f"    [FENCE] YOLO@0.01 failed: {_fc_err}")

                            if _fence_confirmed:
                                break

                        if not _fence_confirmed and show_summary:
                            print(f"    [FENCE] All fence-cleaned strips failed "
                                  f"both ResNet50 and YOLO@0.01 — skipped.")

                    if not _any_strip_confirmed:
                        if show_summary:
                            print(f"    [REMAINDER] {rsrc} (cov={cov:.0%}): "
                                  f"all uncovered sub-regions failed species gate — skipped")
                        continue

            # Fingerprint the half
            try:
                mean_h   = float(np.mean(cv2.cvtColor(half_img, cv2.COLOR_BGR2GRAY)))
                fp_half  = extract_fingerprint(
                    enhance_dark_crop(half_img) if mean_h < 50 else half_img)
            except Exception:
                continue

            # Similarity guard: compare against every tiger already detected
            # in THIS image. Prefer DB fingerprint (running average — more
            # stable than a single crop) so well-established tigers get a
            # higher similarity score and are less likely to be double-counted.
            max_sim_to_existing = 0.0
            for existing_det in result["detections"]:
                tid = existing_det.get("tiger_id", "")
                if tid and tid in tiger_db:
                    # Use DB running-average fingerprint — much more stable
                    fp_ex = tiger_db[tid]
                else:
                    # Fallback: extract fingerprint from the detection crop
                    ex1, ey1, ex2, ey2 = existing_det["bbox"]
                    ex_crop = img_proc[max(0, ey1 - CROP_PAD): min(ih, ey2 + CROP_PAD),
                                       max(0, ex1 - CROP_PAD): min(iw, ex2 + CROP_PAD)]
                    if ex_crop.size == 0:
                        continue
                    try:
                        mean_ec = float(np.mean(cv2.cvtColor(ex_crop, cv2.COLOR_BGR2GRAY)))
                        fp_ex   = extract_fingerprint(
                            enhance_dark_crop(ex_crop) if mean_ec < 50 else ex_crop)
                    except Exception:
                        continue
                try:
                    sim = float(_cos_sim(fp_half.reshape(1, -1),
                                        fp_ex.reshape(1, -1))[0][0])
                    max_sim_to_existing = max(max_sim_to_existing, sim)
                except Exception:
                    pass

            if max_sim_to_existing >= _rem_sim_thresh:
                if show_summary:
                    print(f"    [REMAINDER] {rsrc}: same tiger as existing detection "
                          f"(sim={max_sim_to_existing:.2f}, thresh={_rem_sim_thresh:.2f}) — skipped")
                continue

            # ── STANDING TIGER UPPER-BODY GUARD — Remainder_T only (Bug Fix 15) ──
            # PROBLEM: A walking/standing tiger's YOLO box captures the torso +
            # legs but the head/back above the box spills into the top remainder.
            # The top half passes species gate (head + stripe visible) and has
            # similarity ~0.45–0.60 to the main detection — below 0.82 — so it
            # gets enrolled as a new tiger. Same class of bug as the sitting tiger
            # lower-body guard (Bug Fix 13), but in reverse direction.
            #
            # Observed in 000976.jpg (Test Data 10): single tiger walking, YOLO
            # box captures center body, Remainder_T finds head+back at sim=0.57
            # → falsely enrolled as Tiger_003.
            #
            # FIX: For Remainder_T only, check:
            #   1. An existing detection covers > 40% of image height (tiger is
            #      large in frame — its body extends into the top strip).
            #   2. That existing box extends into the top half (top edge < midpoint).
            #   3. Candidate similarity > 0.30 (still recognizably same species).
            if rsrc == "Remainder_T":
                _standing_guard_triggered = False
                for _edet in result["detections"]:
                    _ex1, _ey1, _ex2, _ey2 = _edet["bbox"]
                    _det_h_ratio = (_ey2 - _ey1) / ih
                    _extends_into_top = _ey1 < (ih // 2)
                    if (_det_h_ratio > 0.40
                            and _extends_into_top
                            and max_sim_to_existing > 0.30):
                        _standing_guard_triggered = True
                        if show_summary:
                            print(
                                f"    [REMAINDER] {rsrc}: STANDING TIGER guard — "
                                f"existing box covers {_det_h_ratio:.0%} of frame height "
                                f"and extends into top half, candidate sim="
                                f"{max_sim_to_existing:.2f} > 0.30 → upper body of "
                                f"standing tiger, NOT a new individual")
                        break
                if _standing_guard_triggered:
                    continue   # skip enrollment — this is the same animal's upper body

            # ── SITTING TIGER LOWER-BODY GUARD — Remainder_B only (Bug Fix 13) ──
            # PROBLEM: A sitting tiger's YOLO box captures the head + torso but
            # misses the haunches / folded legs below the box. The bottom remainder
            # scan finds the haunches, which:
            #   • Pass the species gate (stripe patterns are still visible)
            #   • Have LOW similarity to the main detection (~0.35–0.45) because
            #     upper-body texture (head, back stripes) differs from lower-body
            #     texture (haunches, belly, folded legs).
            # Result: haunches are below REMAINDER_SAME_TIGER_THRESH (0.82) and
            # get enrolled as a new tiger — a false positive.
            #
            # FIX: For Remainder_B only, check two conditions:
            #   1. An existing detection already covers > 50% of the image height.
            #      A sitting tiger is tall in frame — its YOLO box spans more than
            #      half the image vertically. A walking tiger whose cub is genuinely
            #      at the bottom typically has a shorter, wider box (< 50% height).
            #   2. That existing box extends into the bottom half (bottom edge >
            #      image midpoint). This confirms the animal is already tall in frame
            #      and the "uncovered" strip below is just its lower body, not a gap
            #      followed by a second animal.
            #   3. The candidate's similarity to the existing tiger is > 0.30.
            #      If it's truly a different individual (e.g., a cub with very
            #      different stripe colouring), similarity would be < 0.30. Above
            #      0.30 means the ResNet50 still sees the same species/individual.
            #
            # WHY Remainder_B ONLY: side-by-side animals (L/R) are NOT affected
            # by sitting posture — those are handled by the cross-half guard.
            # Top remainder (T) is the head end — sitting or standing, the head
            # is always captured by YOLO and the top strip would fail the species
            # gate anyway. Only the bottom strip is the problem case.
            if rsrc == "Remainder_B":
                _sitting_guard_triggered = False
                for _edet in result["detections"]:
                    _ex1, _ey1, _ex2, _ey2 = _edet["bbox"]
                    _det_h_ratio = (_ey2 - _ey1) / ih
                    _extends_into_bottom = _ey2 > (ih // 2)
                    if (_det_h_ratio > 0.50
                            and _extends_into_bottom
                            and max_sim_to_existing > 0.30):
                        _sitting_guard_triggered = True
                        if show_summary:
                            print(
                                f"    [REMAINDER] {rsrc}: SITTING TIGER guard — "
                                f"existing box covers {_det_h_ratio:.0%} of frame height "
                                f"and extends into bottom half, candidate sim="
                                f"{max_sim_to_existing:.2f} > 0.30 → lower body of "
                                f"sitting tiger, NOT a new individual")
                        break
                if _sitting_guard_triggered:
                    continue   # skip enrollment — this is the same animal's lower body

            # ── WATER REFLECTION GUARD — all remainder halves (Bug Fix 14) ──
            # PROBLEM: A tiger walking beside water produces a mirror reflection
            # in the water surface. The reflection occupies a different region
            # of the frame (left, right, or bottom half) and:
            #   • Passes the species gate (stripe patterns visible in reflection)
            #   • Has similarity ~0.45–0.60 to the real tiger — below 0.82
            # → Remainder scan enrolls the reflection as a second tiger.
            #
            # FIX: Compare this half against the OPPOSITE half (not the tight
            # YOLO crop — background dilutes the crop comparison signal).
            #
            #   direct_sim = similarity(this_half, opposite_half)
            #   flip_sim   = similarity(flipped_this_half, opposite_half)
            #
            # TRIGGER: flip_sim > direct_sim  AND  flip_sim >= 0.55
            #
            # Why the 0.55 absolute minimum?
            #   detect_water_presence() misses brown/muddy water (non-blue).
            #   Removing the water-presence gate would fire for night images
            #   where overall similarity is low and flip_sim ≈ 0.42.
            #   Real reflections have clear flip_sim ≥ 0.65 (calibrated on
            #   007004, 007010). Night-image noise stays below 0.55.
            #   The relative check (flip_sim > direct_sim) is still required
            #   to distinguish reflection from a second tiger facing the
            #   opposite direction.
            #
            #   Flip axis by remainder source:
            #     Remainder_L / Remainder_R → horizontal flip (axis=1)
            #     Remainder_T / Remainder_B → vertical   flip (axis=0)
            REFL_ABS_MIN = 0.55   # minimum flip_sim to rule out night-image noise
            _refl_axis = 1 if rsrc in ("Remainder_L", "Remainder_R") else 0
            # Build the opposite half to compare against
            if rsrc == "Remainder_L":
                _opp_half = img_proc[:, mid_x:]
            elif rsrc == "Remainder_R":
                _opp_half = img_proc[:, :mid_x]
            elif rsrc == "Remainder_T":
                _opp_half = img_proc[mid_y:, :]
            else:  # Remainder_B
                _opp_half = img_proc[:mid_y, :]
            try:
                _fp_opp        = extract_fingerprint(_opp_half)
                # direct_sim: how similar is this half to the opposite half
                _direct_sim    = float(_cos_sim(
                    fp_half.reshape(1, -1), _fp_opp.reshape(1, -1))[0][0])
                # flip_sim: similarity of FLIPPED this half to the opposite.
                # A real reflection, when flipped back, looks MORE like the
                # opposite (it un-mirrors itself). A genuine second tiger
                # does not become more similar when flipped.
                _flipped_half  = cv2.flip(half_img, _refl_axis)
                _fp_flip       = extract_fingerprint(_flipped_half)
                _flip_sim_rem  = float(_cos_sim(
                    _fp_flip.reshape(1, -1), _fp_opp.reshape(1, -1))[0][0])
                _refl_confirmed = (
                    _flip_sim_rem > _direct_sim          # relative: flipped is more similar
                    and _flip_sim_rem >= REFL_ABS_MIN    # absolute: strong enough signal
                )
                if show_summary:
                    if _refl_confirmed:
                        print(
                            f"    [REMAINDER] {rsrc}: WATER REFLECTION — "
                            f"flip_sim={_flip_sim_rem:.2f} > direct_sim={_direct_sim:.2f}"
                            f" (min={REFL_ABS_MIN}) → skipped")
                    else:
                        print(
                            f"    [REMAINDER] {rsrc}: reflection check: "
                            f"flip_sim={_flip_sim_rem:.2f}, direct_sim={_direct_sim:.2f}"
                            f" → NOT a reflection, proceeding")
            except Exception:
                _refl_confirmed = False
            if _refl_confirmed:
                continue   # skip — reflection of the real tiger, not a new individual

            # Cross-half guard: compare against tigers that were JUST enrolled
            # from a previous remainder half in this same image. Uses a lower
            # threshold (0.68) because L/R half-views of the same tiger score
            # ~0.72–0.77 — below the 0.82 YOLO guard but above 0.68.
            # Adult vs cub scores ~0.39, so legitimate pairs are not blocked.
            if remainder_enrolled_fps:
                max_sim_cross = max(
                    float(_cos_sim(fp_half.reshape(1, -1), fp_r.reshape(1, -1))[0][0])
                    for fp_r in remainder_enrolled_fps
                )
                if max_sim_cross >= REMAINDER_CROSS_HALF_THRESH:
                    if show_summary:
                        print(f"    [REMAINDER] {rsrc}: same tiger as remainder-enrolled "
                              f"(cross-half sim={max_sim_cross:.2f}) — skipped")
                    continue
                if show_summary and max_sim_cross > 0:
                    print(f"    [REMAINDER] {rsrc}: cross-half sim={max_sim_cross:.2f} "
                          f"< {REMAINDER_CROSS_HALF_THRESH} — proceeding")

            if show_summary:
                print(f"    [REMAINDER] {rsrc}: new tiger confirmed "
                      f"(cov={cov:.0%}, max_sim={max_sim_to_existing:.2f})")

            # Grad-CAM (optional)
            if run_gradcam:
                gcam_path = GRADCAM_DIR / f"gradcam_{img_path.stem}_rem{valid_count+1}.jpg"
                generate_gradcam(half_img, gcam_path,
                                 label=f"Remainder|{rsrc}")

            # Enrol via identity matching
            try:
                fp_enrol = extract_fingerprint(
                    enhance_dark_crop(half_img) if mean_h < 50 else half_img)
                rem_id, rem_score, rem_new = match_or_enroll(fp_enrol, update_db=True)
            except Exception as e:
                if show_summary:
                    print(f"    [REMAINDER] fingerprint failed: {e}")
                continue

            rem_morph  = classify_color_morph(half_img)
            rem_tinfo  = TIGER_TYPES.get(rem_morph, {"display": rem_morph,
                                                      "color": (0, 255, 0)})
            rem_vp     = classify_viewpoint(rx1, ry1, rx2, ry2, iw, ih)
            rem_status = "New Enrollment" if rem_new else f"Recaptured {rem_score:.2f}"

            # Annotate half-frame box on the full image
            cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), rem_tinfo["color"], 3)
            lbl1 = f"{rem_id}  {rem_tinfo['display']}  [{rem_vp}]"
            lbl2 = f"Src:{rsrc[:4]}  Match:{rem_score:.2f}  {rem_status}"
            (lw, lh), _ = cv2.getTextSize(lbl1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            bg_top = max(0, ry1 - lh * 2 - 14)
            cv2.rectangle(annotated, (rx1, bg_top),
                          (rx1 + max(lw + 8, 260), ry1), rem_tinfo["color"], -1)
            tc = (0, 0, 0) if rem_morph in ("White", "Snow_White", "Golden") \
                           else (255, 255, 255)
            cv2.putText(annotated, lbl1, (rx1 + 4, ry1 - lh - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, tc, 2)
            cv2.putText(annotated, lbl2, (rx1 + 4, ry1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, tc, 1)

            valid_count += 1
            remainder_enrolled_fps.append(fp_enrol)        # track for cross-half guard
            accepted_coords.append((rx1, ry1, rx2, ry2))  # prevent overlap in next half
            result["detections"].append({
                "tiger_id"        : rem_id,
                "morph"           : rem_morph,
                "display"         : rem_tinfo["display"],
                "visibility"      : "Full_Body",
                "viewpoint"       : rem_vp,
                "detection_source": rsrc,
                "det_conf"        : 0.0,
                "match_score"     : round(rem_score, 4),
                "is_new"          : rem_new,
                "id_status"       : rem_status,
                "top3_labels"     : " | ".join(rem_top3),
                "bbox"            : (rx1, ry1, rx2, ry2),
            })

            if show_summary:
                print(f"    {rem_id:<12} {rem_tinfo['display']:<22} "
                      f"Full_Body      {rem_vp:<18} "
                      f"match={rem_score:.2f}  {rem_status}  [{rsrc}]")

    # ── MARGIN SCAN ──────────────────────────────────────────────────────────────
    # When a large tiger spans almost the full frame, its YOLO box covers every
    # half above the REMAINDER_COVERAGE_THRESH, so the remainder scan finds
    # nothing. The Margin Scan looks at the STRIPS OUTSIDE the union of all
    # detected bounding boxes. These strips are the only uncovered pixels — any
    # second tiger partially visible at the frame edge will appear there.
    #
    # Calibration: strip must be ≥ MARGIN_MIN_DIM in BOTH dimensions.
    # (004840: bottom margin = 151 px tall × full width — viable ✓)
    MARGIN_MIN_DIM = 60   # pixels; below this a strip is too small to hold a tiger
    # (004840 bottom margin = 77px after CROP_PAD expansion → 77 > 60 ✓)
    MARGIN_SAME_TIGER_THRESH = 0.80   # slightly lower than REMAINDER — partial body
    #   If strip fp is > 0.80 similar to an existing tiger → same animal, skip.
    #   Partial-body strips of the SAME tiger score ~0.82–0.88.
    #   Different individuals score ~0.55–0.70.

    # ── FIX G cont'd: Allow Margin scan even when vertical guard triggered ──
    # Same rationale as Remainder scan — use stricter similarity threshold
    # (0.70 vs 0.80) when vertical to prevent re-enrolling the same animal.
    if valid_count >= 1 and iw >= 80 and ih >= 80:
        from sklearn.metrics.pairwise import cosine_similarity as _cos_sim
        _margin_sim_thresh = 0.70 if vertical_tiger_detected else MARGIN_SAME_TIGER_THRESH

        # Union bounding box of all accepted detections
        ux1 = min(d["bbox"][0] for d in result["detections"])
        uy1 = min(d["bbox"][1] for d in result["detections"])
        ux2 = max(d["bbox"][2] for d in result["detections"])
        uy2 = max(d["bbox"][3] for d in result["detections"])

        # Four candidate margin strips around the union box
        margin_candidates = []
        if ux1 >= MARGIN_MIN_DIM:                  # left strip
            margin_candidates.append(
                (0, 0, ux1, ih, "Margin_L", img_proc[:, :ux1]))
        if (iw - ux2) >= MARGIN_MIN_DIM:           # right strip
            margin_candidates.append(
                (ux2, 0, iw, ih, "Margin_R", img_proc[:, ux2:]))
        if uy1 >= MARGIN_MIN_DIM:                  # top strip
            margin_candidates.append(
                (0, 0, iw, uy1, "Margin_T", img_proc[:uy1, :]))
        if (ih - uy2) >= MARGIN_MIN_DIM:           # bottom strip  ← the key one for 004840
            margin_candidates.append(
                (0, uy2, iw, ih, "Margin_B", img_proc[uy2:, :]))

        margin_enrolled_fps = []   # cross-margin dedup (same logic as remainder)

        for (mx1, my1, mx2, my2, msrc, mstrip) in margin_candidates:
            if mstrip.shape[0] < MARGIN_MIN_DIM or mstrip.shape[1] < MARGIN_MIN_DIM:
                continue

            # Species gate
            m_ok, m_top3 = verify_species_top3(mstrip)
            if not m_ok:
                if show_summary:
                    print(f"    [MARGIN] {msrc}: species rejected — {m_top3}")
                continue

            # ── Bug Fix 20b: Narrow margin top-5-only guard ─────────────
            # A narrow margin strip (< 100px) that only passed species gate
            # via top-5 widening is unreliable — top-5 catches false positives
            # like "hen-of-the-woods" ranked #4 alongside "tiger" at #5.
            # For narrow strips, require top-3 confirmation (no "(top5)" tag).
            # Observed: 002627.jpg Margin_R (76px wide) — fence mesh strip
            # passed via top-5, enrolled as false Tiger_003.
            _narrow_dim = min(mstrip.shape[0], mstrip.shape[1])
            if _narrow_dim < 100 and any("(top5)" in str(l) for l in m_top3):
                if show_summary:
                    print(f"    [MARGIN] {msrc}: narrow strip ({_narrow_dim}px) "
                          f"only passed via top-5 widening ({m_top3}) — skipped")
                continue

            # Fingerprint
            try:
                mean_ms = float(np.mean(cv2.cvtColor(mstrip, cv2.COLOR_BGR2GRAY)))
                fp_margin = extract_fingerprint(
                    enhance_dark_crop(mstrip) if mean_ms < 50 else mstrip)
            except Exception:
                continue

            # Similarity guard vs. existing detections (using DB fingerprint)
            max_sim_m = 0.0
            for existing_det in result["detections"]:
                tid = existing_det.get("tiger_id", "")
                if tid and tid in tiger_db:
                    fp_ex = tiger_db[tid]
                else:
                    ex1, ey1, ex2, ey2 = existing_det["bbox"]
                    ex_crop = img_proc[max(0, ey1 - CROP_PAD): min(ih, ey2 + CROP_PAD),
                                       max(0, ex1 - CROP_PAD): min(iw, ex2 + CROP_PAD)]
                    if ex_crop.size == 0:
                        continue
                    try:
                        mean_ec = float(np.mean(cv2.cvtColor(ex_crop, cv2.COLOR_BGR2GRAY)))
                        fp_ex = extract_fingerprint(
                            enhance_dark_crop(ex_crop) if mean_ec < 50 else ex_crop)
                    except Exception:
                        continue
                try:
                    sim = float(_cos_sim(fp_margin.reshape(1, -1),
                                        fp_ex.reshape(1, -1))[0][0])
                    max_sim_m = max(max_sim_m, sim)
                except Exception:
                    pass

            if max_sim_m >= _margin_sim_thresh:
                if show_summary:
                    print(f"    [MARGIN] {msrc}: same tiger as existing "
                          f"(sim={max_sim_m:.2f}, thresh={_margin_sim_thresh:.2f}) — skipped")
                continue

            # ── WATER REFLECTION GUARD — margin strips (Bug Fix 14b) ──
            # Same criterion as the remainder scan reflection guard:
            #   flip_sim > direct_sim  AND  flip_sim >= REFL_ABS_MIN (0.55)
            # Compare the margin strip against the OPPOSITE IMAGE HALF (not the
            # YOLO-box-relative area — that can be too thin when the box is near
            # the edge, causing the guard to be silently skipped).
            # Using fixed image halves guarantees a comparison region ≥ half the
            # image, which is always large enough for a meaningful fingerprint.
            # Axis: L/R margins → horizontal flip (axis=1); T/B → vertical (axis=0)
            _m_refl_axis = 1 if msrc in ("Margin_L", "Margin_R") else 0
            if msrc == "Margin_L":
                _m_opp = img_proc[:, mid_x:]   # right half
            elif msrc == "Margin_R":
                _m_opp = img_proc[:, :mid_x]   # left half
            elif msrc == "Margin_T":
                _m_opp = img_proc[mid_y:, :]   # bottom half
            else:  # Margin_B
                _m_opp = img_proc[:mid_y, :]   # top half (always ≥ half image height)
            _m_refl_confirmed = False
            if _m_opp.shape[0] >= MARGIN_MIN_DIM and _m_opp.shape[1] >= MARGIN_MIN_DIM:
                try:
                    _fp_m_opp   = extract_fingerprint(_m_opp)
                    _m_dir_sim  = float(_cos_sim(
                        fp_margin.reshape(1, -1), _fp_m_opp.reshape(1, -1))[0][0])
                    _m_flipped  = cv2.flip(mstrip, _m_refl_axis)
                    _fp_m_flip  = extract_fingerprint(_m_flipped)
                    _m_flip_sim = float(_cos_sim(
                        _fp_m_flip.reshape(1, -1), _fp_m_opp.reshape(1, -1))[0][0])
                    _m_refl_confirmed = (
                        _m_flip_sim > _m_dir_sim
                        and _m_flip_sim >= REFL_ABS_MIN    # same 0.55 threshold
                    )
                    if show_summary:
                        if _m_refl_confirmed:
                            print(
                                f"    [MARGIN] {msrc}: WATER REFLECTION — "
                                f"flip_sim={_m_flip_sim:.2f} > direct_sim={_m_dir_sim:.2f}"
                                f" (min={REFL_ABS_MIN}) → skipped")
                        else:
                            print(
                                f"    [MARGIN] {msrc}: reflection check: "
                                f"flip_sim={_m_flip_sim:.2f}, direct_sim={_m_dir_sim:.2f}"
                                f" → NOT a reflection, proceeding")
                except Exception:
                    pass
            if _m_refl_confirmed:
                continue   # skip — margin strip is a water reflection

            # Cross-margin dedup (same logic as remainder cross-half guard)
            if margin_enrolled_fps:
                max_sim_cross_m = max(
                    float(_cos_sim(fp_margin.reshape(1, -1),
                                   fp_r.reshape(1, -1))[0][0])
                    for fp_r in margin_enrolled_fps
                )
                if max_sim_cross_m >= REMAINDER_CROSS_HALF_THRESH:
                    if show_summary:
                        print(f"    [MARGIN] {msrc}: same as margin-enrolled "
                              f"(sim={max_sim_cross_m:.2f}) — skipped")
                    continue

            if show_summary:
                print(f"    [MARGIN] {msrc}: new tiger found "
                      f"(strip={mx2-mx1}x{my2-my1}, max_sim={max_sim_m:.2f})")

            # Grad-CAM (optional)
            if run_gradcam:
                gcam_path = GRADCAM_DIR / f"gradcam_{img_path.stem}_margin{valid_count+1}.jpg"
                generate_gradcam(mstrip, gcam_path, label=f"Margin|{msrc}")

            # Enrol via identity matching
            try:
                fp_enrol_m = extract_fingerprint(
                    enhance_dark_crop(mstrip) if mean_ms < 50 else mstrip)
                m_id, m_score, m_new = match_or_enroll(fp_enrol_m, update_db=True)
            except Exception as e:
                if show_summary:
                    print(f"    [MARGIN] enroll failed: {e}")
                continue

            margin_enrolled_fps.append(fp_enrol_m)

            m_morph  = classify_color_morph(mstrip)
            m_tinfo  = TIGER_TYPES.get(m_morph, {"display": m_morph, "color": (0, 200, 100)})
            m_vp     = classify_viewpoint(mx1, my1, mx2, my2, iw, ih)
            m_status = "New Enrollment" if m_new else f"Recaptured {m_score:.2f}"

            # Annotate margin box on the full image
            cv2.rectangle(annotated, (mx1, my1), (mx2, my2), m_tinfo["color"], 3)
            mlbl1 = f"{m_id}  {m_tinfo['display']}  [{m_vp}]"
            mlbl2 = f"Src:{msrc[:3]}  Match:{m_score:.2f}  {m_status}"
            (mlw, mlh), _ = cv2.getTextSize(mlbl1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            mbg_top = max(0, my1 + 4)
            cv2.rectangle(annotated, (mx1, mbg_top),
                          (mx1 + max(mlw + 8, 260), mbg_top + mlh * 2 + 10),
                          m_tinfo["color"], -1)
            mtc = (0, 0, 0) if m_morph in ("White", "Snow_White", "Golden") \
                            else (255, 255, 255)
            cv2.putText(annotated, mlbl1, (mx1 + 4, mbg_top + mlh + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, mtc, 2)
            cv2.putText(annotated, mlbl2, (mx1 + 4, mbg_top + mlh * 2 + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, mtc, 1)

            valid_count += 1
            result["detections"].append({
                "tiger_id"        : m_id,
                "morph"           : m_morph,
                "display"         : m_tinfo["display"],
                "visibility"      : "Partial_Body",
                "viewpoint"       : m_vp,
                "detection_source": msrc,
                "det_conf"        : 0.0,
                "match_score"     : round(m_score, 4),
                "is_new"          : m_new,
                "id_status"       : m_status,
                "top3_labels"     : " | ".join(m_top3),
                "bbox"            : (mx1, my1, mx2, my2),
            })

            if show_summary:
                print(f"    {m_id:<12} {m_tinfo['display']:<22} "
                      f"Partial_Body   {m_vp:<18} "
                      f"match={m_score:.2f}  {m_status}  [{msrc}]")

    # ── FENCE-ZONE TIGER SCAN ─────────────────────────────────────────────────────
    # Triggered ONLY when a fence pattern is confirmed in the image.
    #
    # Problem: a tiger on the OTHER SIDE of a fence is missed by every scan above
    # because:
    #   • Remainder scan skips the half containing the fence tiger: cov > 65%
    #     (the main tiger's large YOLO box covers most of that half)
    #   • Margin scan strip is too narrow (< 60px) because the main box extends
    #     nearly to the fence
    #   • Strip-gate fence retry also fails: the strip is too narrow for both
    #     ResNet50 and YOLO@0.01 to confirm a tiger body
    #
    # Fix: partition the image into LEFT and RIGHT zones based on the union
    # bounding box of all existing detections, with a 20% overlap buffer to
    # capture the full fence-tiger body (not just the uncovered edge).
    # On each zone: (1) remove fence bars, (2) run YOLO at conf=0.01.
    # Any new tiger box that clears all guards → enroll as a second individual.
    #
    # Safety gates (same strictness as LOWCONF):
    #   (a) IoU with every existing detection < 0.30 → genuinely new region
    #   (b) Species gate (ResNet50) on fence-cleaned crop → confirmed tiger
    #   (c) Cosine similarity to existing tigers < 0.82 → different individual
    #
    # Ground truth calibration: 000341.jpg — 2 tigers, one behind a fence on
    # the right side. Main tiger box ends near the fence; fence tiger is in the
    # zone to the right of that box.
    # ── FIX A: Allow Fence-Zone scan even when vertical guard triggered ──
    # (Bug Fix 2026-06-10) Fence-Zone scans OUTSIDE the main detection box,
    # so the first tiger's vertical pose is irrelevant to finding a second
    # tiger behind a fence. Removing `not vertical_tiger_detected` gate.
    if (valid_count >= 1
            and oiv7_detector is not None
            and detect_fence_pattern(img_proc)):

        if show_summary:
            print(f"\n    [FENCE-ZONE] Fence confirmed — scanning zones outside main detection.")

        from sklearn.metrics.pairwise import cosine_similarity as _cos_sim

        _fz_accepted = [d["bbox"] for d in result["detections"]]
        _fz_ux1 = min(b[0] for b in _fz_accepted)
        _fz_ux2 = max(b[2] for b in _fz_accepted)

        # Overlap buffer: 20% of image width, so the zone starts slightly inside
        # the main detection area. This lets YOLO build a coherent bounding box
        # even if the fence tiger's body partially overlaps the main tiger region.
        _fz_overlap = int(iw * 0.20)
        _FENCE_ZONE_FP_BAD = {"tiger cat", "tiger shark", "tiger beetle"}

        # SIGNAL TRACKER: only set True when YOLO finds a tiger-class box in the
        # fence zone that is NOT the same box as an existing detection (IoU < 0.60).
        # If this stays False after all scans, it means NOTHING tiger-like is visible
        # behind the fence → no reason to ask a human. We only flag when the model
        # finds an ambiguous signal it cannot cleanly resolve.
        _fz_tiger_signal_found = False

        fence_zones = []
        if _fz_ux1 >= 30:                                      # left zone viable
            fence_zones.append((0, 0, min(iw, _fz_ux1 + _fz_overlap), ih, "FenceZone_L"))
        if (iw - _fz_ux2) >= 30:                              # right zone viable
            fence_zones.append((max(0, _fz_ux2 - _fz_overlap), 0, iw, ih, "FenceZone_R"))

        for (fzx1, fzy1, fzx2, fzy2, fzsrc) in fence_zones:
            fz_crop = img_proc[fzy1:fzy2, fzx1:fzx2]
            if fz_crop.shape[0] < 30 or fz_crop.shape[1] < 30:
                continue

            fz_clean = remove_fence_bars(fz_crop)

            try:
                fz_res = oiv7_detector(fz_clean, verbose=False,
                                       conf=0.01, iou=0.45)[0]
            except Exception as _fze:
                if show_summary:
                    print(f"    [FENCE-ZONE] {fzsrc}: YOLO failed — {_fze}")
                continue

            if fz_res.boxes is None or len(fz_res.boxes) == 0:
                if show_summary:
                    print(f"    [FENCE-ZONE] {fzsrc}: no detections")
                continue

            _fz_found_new = False
            for fz_box, fz_cls, fz_conf_val in zip(
                    fz_res.boxes.xyxy.cpu().numpy(),
                    fz_res.boxes.cls.cpu().numpy(),
                    fz_res.boxes.conf.cpu().numpy()):

                fz_lbl   = oiv7_detector.names[int(fz_cls)]
                fz_lbl_l = fz_lbl.lower()
                if not ("tiger" in fz_lbl_l and
                        not any(fp in fz_lbl_l for fp in _FENCE_ZONE_FP_BAD)):
                    continue

                # Translate to full-image coordinates with padding
                fbx1 = max(0,  fzx1 + int(fz_box[0]) - CROP_PAD)
                fby1 = max(0,  fzy1 + int(fz_box[1]) - CROP_PAD)
                fbx2 = min(iw, fzx1 + int(fz_box[2]) + CROP_PAD)
                fby2 = min(ih, fzy1 + int(fz_box[3]) + CROP_PAD)
                if fbx2 <= fbx1 or fby2 <= fby1:
                    continue

                # (a) IoU guard — must be genuinely new region
                fz_max_iou = 0.0
                fb_area = max(1, (fbx2 - fbx1) * (fby2 - fby1))
                for (ax1, ay1, ax2, ay2) in _fz_accepted:
                    ix1 = max(fbx1, ax1); iy1 = max(fby1, ay1)
                    ix2 = min(fbx2, ax2); iy2 = min(fby2, ay2)
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    if inter == 0:
                        continue
                    union = fb_area + (ax2 - ax1) * (ay2 - ay1) - inter
                    fz_max_iou = max(fz_max_iou, inter / max(1, union))

                # Mark signal: YOLO found a tiger-class box that is NOT the exact
                # same box as an existing detection (IoU < 0.60). This is real
                # evidence of something tiger-like in the fence zone, even if we
                # ultimately can't confirm or enroll it.
                if fz_max_iou < 0.60:
                    _fz_tiger_signal_found = True

                if fz_max_iou >= 0.30:
                    if show_summary:
                        print(f"    [FENCE-ZONE] {fzsrc} box ({fbx1},{fby1})-({fbx2},{fby2}) "
                              f"conf={fz_conf_val:.3f}: overlaps existing "
                              f"(IoU={fz_max_iou:.2f}) — ambiguous, cannot enroll")
                    continue

                # (b) Species gate on fence-cleaned crop at actual image coordinates
                fz_real_crop = img_proc[fby1:fby2, fbx1:fbx2]
                if fz_real_crop.shape[0] < 20 or fz_real_crop.shape[1] < 20:
                    continue
                fz_spec_ok, fz_top3 = verify_species_top3(remove_fence_bars(fz_real_crop))
                if not fz_spec_ok:
                    if show_summary:
                        print(f"    [FENCE-ZONE] {fzsrc} box ({fbx1},{fby1})-({fbx2},{fby2}) "
                              f"conf={fz_conf_val:.3f}: species rejected — {fz_top3}")
                    continue

                # (c) Fingerprint + cosine similarity guard
                try:
                    mean_fz = float(np.mean(cv2.cvtColor(fz_real_crop, cv2.COLOR_BGR2GRAY)))
                    fp_fz   = extract_fingerprint(
                        enhance_dark_crop(fz_real_crop) if mean_fz < 50 else fz_real_crop)
                except Exception:
                    continue

                fz_max_sim = 0.0
                for existing_det in result["detections"]:
                    tid = existing_det.get("tiger_id", "")
                    if tid and tid in tiger_db:
                        fp_ex = tiger_db[tid]
                    else:
                        ex1, ey1, ex2, ey2 = existing_det["bbox"]
                        ec = img_proc[max(0, ey1-CROP_PAD): min(ih, ey2+CROP_PAD),
                                      max(0, ex1-CROP_PAD): min(iw, ex2+CROP_PAD)]
                        if ec.size == 0:
                            continue
                        try:
                            mean_ec = float(np.mean(cv2.cvtColor(ec, cv2.COLOR_BGR2GRAY)))
                            fp_ex   = extract_fingerprint(
                                enhance_dark_crop(ec) if mean_ec < 50 else ec)
                        except Exception:
                            continue
                    try:
                        sim = float(_cos_sim(fp_fz.reshape(1, -1),
                                             fp_ex.reshape(1, -1))[0][0])
                        fz_max_sim = max(fz_max_sim, sim)
                    except Exception:
                        pass

                if fz_max_sim >= 0.82:
                    if show_summary:
                        print(f"    [FENCE-ZONE] {fzsrc}: same tiger as existing "
                              f"(sim={fz_max_sim:.2f}) — skipped")
                    continue

                # ── Bug Fix 17: FenceZone high-similarity soft-block ──────
                # When a FenceZone detection has similarity 0.65–0.82 to an
                # existing tiger, it's ambiguous — could be the same animal
                # seen through vegetation/fence bars, or a genuine 2nd tiger
                # that looks similar. Don't auto-enroll; flag for human review.
                # Observed: 000985.jpg FenceZone_R found same tiger through
                # branches at sim=0.70 → false Tiger_004 enrollment.
                FENCEZONE_AMBIGUOUS_THRESH = 0.65
                if fz_max_sim >= FENCEZONE_AMBIGUOUS_THRESH:
                    if show_summary:
                        print(f"    [FENCE-ZONE] {fzsrc} box ({fbx1},{fby1})-({fbx2},{fby2}) "
                              f"conf={fz_conf_val:.3f}: overlaps existing "
                              f"(IoU={fz_max_iou:.2f}) — ambiguous, cannot enroll")
                    continue

                if show_summary:
                    print(f"    [FENCE-ZONE] {fzsrc}: NEW tiger found! "
                          f"box=({fbx1},{fby1})-({fbx2},{fby2}) "
                          f"conf={fz_conf_val:.3f}  IoU={fz_max_iou:.2f}  "
                          f"sim={fz_max_sim:.2f}")

                # Enroll
                try:
                    fp_fz_enrol = extract_fingerprint(
                        enhance_dark_crop(fz_real_crop) if mean_fz < 50 else fz_real_crop)
                    fz_id, fz_score, fz_new = match_or_enroll(fp_fz_enrol, update_db=True)
                except Exception as e:
                    if show_summary:
                        print(f"    [FENCE-ZONE] enroll failed: {e}")
                    continue

                fz_morph  = classify_color_morph(fz_real_crop)
                fz_tinfo  = TIGER_TYPES.get(fz_morph, {"display": fz_morph,
                                                        "color": (255, 140, 0)})
                fz_vp     = classify_viewpoint(fbx1, fby1, fbx2, fby2, iw, ih)
                fz_status = "New Enrollment" if fz_new else f"Recaptured {fz_score:.2f}"

                cv2.rectangle(annotated, (fbx1, fby1), (fbx2, fby2),
                              fz_tinfo["color"], 3)
                fl1 = f"{fz_id}  {fz_tinfo['display']}  [{fz_vp}]"
                fl2 = (f"Src:FenceZone  Det:{fz_conf_val:.2f}  "
                       f"Match:{fz_score:.2f}  {fz_status}")
                (flw, flh), _ = cv2.getTextSize(fl1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                fbg_top = max(0, fby1 - flh * 2 - 14)
                cv2.rectangle(annotated, (fbx1, fbg_top),
                              (fbx1 + max(flw + 8, 260), fby1),
                              fz_tinfo["color"], -1)
                ftc = ((0, 0, 0) if fz_morph in ("White", "Snow_White", "Golden")
                       else (255, 255, 255))
                cv2.putText(annotated, fl1, (fbx1 + 4, fby1 - flh - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, ftc, 2)
                cv2.putText(annotated, fl2, (fbx1 + 4, fby1 - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, ftc, 1)

                _fz_accepted.append((fbx1, fby1, fbx2, fby2))
                valid_count += 1
                result["detections"].append({
                    "tiger_id"        : fz_id,
                    "morph"           : fz_morph,
                    "display"         : fz_tinfo["display"],
                    "visibility"      : "Partial_Body",
                    "viewpoint"       : fz_vp,
                    "detection_source": "FenceZone_OIV7",
                    "det_conf"        : round(float(fz_conf_val), 4),
                    "match_score"     : round(fz_score, 4),
                    "is_new"          : fz_new,
                    "id_status"       : fz_status,
                    "top3_labels"     : " | ".join(fz_top3),
                    "bbox"            : (fbx1, fby1, fbx2, fby2),
                })

                if show_summary:
                    print(f"    {fz_id:<12} {fz_tinfo['display']:<22} "
                          f"Partial_Body   {fz_vp:<18} "
                          f"match={fz_score:.2f}  {fz_status}  [FenceZone_OIV7]")

                _fz_found_new = True
                break   # one fence-zone tiger per zone; move to next zone

            if not _fz_found_new and show_summary:
                print(f"    [FENCE-ZONE] {fzsrc}: no new tiger passed all guards")

        # ── FENCE MASK-SEARCH: last resort ────────────────────────────────────
        # When all fence-zone scans fail, the second tiger is probably SPATIALLY
        # OVERLAPPING the main tiger in the frame (one is in the foreground, the
        # other behind the fence in the background at the same x/y position).
        # YOLO's NMS always suppresses the background box because the foreground
        # box has higher confidence and high IoU with it.
        #
        # Fix: create a masked copy of the image where every accepted detection box
        # is filled with the median colour of the surrounding border. This removes
        # the foreground tiger's texture that confuses YOLO, allowing it to "see"
        # the background tiger body that was hidden underneath.
        #
        # Only runs when the fence zone scan found nothing new (valid_count still == 1).
        if valid_count == 1 and oiv7_detector is not None:
            _fz_mask_img = img_proc.copy()
            for (ax1, ay1, ax2, ay2) in _fz_accepted:
                # Fill box with surrounding border median — less jarring than black
                border_pixels = []
                pad = 8
                if ay1 - pad >= 0:
                    border_pixels.append(_fz_mask_img[max(0, ay1-pad):ay1, ax1:ax2])
                if ay2 + pad <= ih:
                    border_pixels.append(_fz_mask_img[ay2:min(ih, ay2+pad), ax1:ax2])
                if ax1 - pad >= 0:
                    border_pixels.append(_fz_mask_img[ay1:ay2, max(0, ax1-pad):ax1])
                if ax2 + pad <= iw:
                    border_pixels.append(_fz_mask_img[ay1:ay2, ax2:min(iw, ax2+pad)])
                if border_pixels:
                    all_pixels = np.concatenate(
                        [p.reshape(-1, 3) for p in border_pixels], axis=0)
                    fill_color = tuple(int(v) for v in np.median(all_pixels, axis=0))
                else:
                    fill_color = (128, 128, 128)
                _fz_mask_img[ay1:ay2, ax1:ax2] = fill_color

            _fz_clean_mask = remove_fence_bars(_fz_mask_img)

            try:
                fz_mask_res = oiv7_detector(_fz_clean_mask, verbose=False,
                                            conf=0.01, iou=0.45)[0]
            except Exception as _fzme:
                fz_mask_res = None
                if show_summary:
                    print(f"    [FENCE-MASK] YOLO on masked image failed: {_fzme}")

            if fz_mask_res is not None and fz_mask_res.boxes is not None:
                for fzm_box, fzm_cls, fzm_conf in zip(
                        fz_mask_res.boxes.xyxy.cpu().numpy(),
                        fz_mask_res.boxes.cls.cpu().numpy(),
                        fz_mask_res.boxes.conf.cpu().numpy()):
                    fzm_lbl   = oiv7_detector.names[int(fzm_cls)]
                    fzm_lbl_l = fzm_lbl.lower()
                    if not ("tiger" in fzm_lbl_l and
                            not any(fp in fzm_lbl_l for fp in _FENCE_ZONE_FP_BAD)):
                        continue

                    mx1 = max(0,  int(fzm_box[0]) - CROP_PAD)
                    my1 = max(0,  int(fzm_box[1]) - CROP_PAD)
                    mx2 = min(iw, int(fzm_box[2]) + CROP_PAD)
                    my2 = min(ih, int(fzm_box[3]) + CROP_PAD)
                    if mx2 <= mx1 or my2 <= my1:
                        continue

                    fzm_max_iou = 0.0
                    fzm_area = max(1, (mx2 - mx1) * (my2 - my1))
                    for (ax1, ay1, ax2, ay2) in _fz_accepted:
                        ix1 = max(mx1, ax1); iy1 = max(my1, ay1)
                        ix2 = min(mx2, ax2); iy2 = min(my2, ay2)
                        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                        if inter == 0:
                            continue
                        union = fzm_area + (ax2 - ax1) * (ay2 - ay1) - inter
                        fzm_max_iou = max(fzm_max_iou, inter / max(1, union))

                    # Mark signal: mask-search found a tiger-class box distinct
                    # from any existing detection → something is there
                    if fzm_max_iou < 0.60:
                        _fz_tiger_signal_found = True

                    if fzm_max_iou >= 0.60:
                        if show_summary:
                            print(f"    [FENCE-MASK] box ({mx1},{my1})-({mx2},{my2}) "
                                  f"conf={fzm_conf:.3f}: IoU={fzm_max_iou:.2f} — skipped")
                        continue

                    # Species gate on original image crop (not the masked version)
                    fzm_real_crop = img_proc[my1:my2, mx1:mx2]
                    if fzm_real_crop.shape[0] < 20 or fzm_real_crop.shape[1] < 20:
                        continue
                    fzm_spec_ok, fzm_top3 = verify_species_top3(
                        remove_fence_bars(fzm_real_crop))
                    if not fzm_spec_ok:
                        if show_summary:
                            print(f"    [FENCE-MASK] ({mx1},{my1})-({mx2},{my2}) "
                                  f"species rejected — {fzm_top3}")
                        continue

                    # Cosine similarity guard
                    try:
                        mean_fzm = float(np.mean(
                            cv2.cvtColor(fzm_real_crop, cv2.COLOR_BGR2GRAY)))
                        fp_fzm = extract_fingerprint(
                            enhance_dark_crop(fzm_real_crop) if mean_fzm < 50
                            else fzm_real_crop)
                    except Exception:
                        continue

                    fzm_max_sim = 0.0
                    for existing_det in result["detections"]:
                        tid = existing_det.get("tiger_id", "")
                        if tid and tid in tiger_db:
                            fp_ex = tiger_db[tid]
                        else:
                            ex1, ey1, ex2, ey2 = existing_det["bbox"]
                            ec = img_proc[max(0,ey1-CROP_PAD):min(ih,ey2+CROP_PAD),
                                         max(0,ex1-CROP_PAD):min(iw,ex2+CROP_PAD)]
                            if ec.size == 0:
                                continue
                            try:
                                mean_ec = float(np.mean(
                                    cv2.cvtColor(ec, cv2.COLOR_BGR2GRAY)))
                                fp_ex = extract_fingerprint(
                                    enhance_dark_crop(ec) if mean_ec < 50 else ec)
                            except Exception:
                                continue
                        try:
                            sim = float(_cos_sim(fp_fzm.reshape(1, -1),
                                                 fp_ex.reshape(1, -1))[0][0])
                            fzm_max_sim = max(fzm_max_sim, sim)
                        except Exception:
                            pass

                    if fzm_max_sim >= 0.82:
                        if show_summary:
                            print(f"    [FENCE-MASK] ({mx1},{my1})-({mx2},{my2}): "
                                  f"same tiger (sim={fzm_max_sim:.2f}) — skipped")
                        continue

                    if show_summary:
                        print(f"    [FENCE-MASK] NEW tiger found via mask search! "
                              f"box=({mx1},{my1})-({mx2},{my2}) "
                              f"conf={fzm_conf:.3f}  IoU={fzm_max_iou:.2f}  "
                              f"sim={fzm_max_sim:.2f}")

                    try:
                        fp_fzm_enrol = extract_fingerprint(
                            enhance_dark_crop(fzm_real_crop) if mean_fzm < 50
                            else fzm_real_crop)
                        fzm_id, fzm_score, fzm_new = match_or_enroll(
                            fp_fzm_enrol, update_db=True)
                    except Exception as e:
                        if show_summary:
                            print(f"    [FENCE-MASK] enroll failed: {e}")
                        continue

                    fzm_morph  = classify_color_morph(fzm_real_crop)
                    fzm_tinfo  = TIGER_TYPES.get(fzm_morph, {
                        "display": fzm_morph, "color": (255, 100, 0)})
                    fzm_vp     = classify_viewpoint(mx1, my1, mx2, my2, iw, ih)
                    fzm_status = ("New Enrollment" if fzm_new
                                  else f"Recaptured {fzm_score:.2f}")

                    cv2.rectangle(annotated, (mx1, my1), (mx2, my2),
                                  fzm_tinfo["color"], 3)
                    fm1 = f"{fzm_id}  {fzm_tinfo['display']}  [{fzm_vp}]"
                    fm2 = (f"Src:FenceMask  Det:{fzm_conf:.2f}  "
                           f"Match:{fzm_score:.2f}  {fzm_status}")
                    (fmlw, fmlh), _ = cv2.getTextSize(
                        fm1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    fmbg = max(0, my1 - fmlh * 2 - 14)
                    cv2.rectangle(annotated, (mx1, fmbg),
                                  (mx1 + max(fmlw + 8, 260), my1),
                                  fzm_tinfo["color"], -1)
                    fmtc = ((0, 0, 0)
                            if fzm_morph in ("White", "Snow_White", "Golden")
                            else (255, 255, 255))
                    cv2.putText(annotated, fm1, (mx1 + 4, my1 - fmlh - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, fmtc, 2)
                    cv2.putText(annotated, fm2, (mx1 + 4, my1 - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.36, fmtc, 1)

                    _fz_accepted.append((mx1, my1, mx2, my2))
                    valid_count += 1
                    result["detections"].append({
                        "tiger_id"        : fzm_id,
                        "morph"           : fzm_morph,
                        "display"         : fzm_tinfo["display"],
                        "visibility"      : "Partial_Body",
                        "viewpoint"       : fzm_vp,
                        "detection_source": "FenceMask_OIV7",
                        "det_conf"        : round(float(fzm_conf), 4),
                        "match_score"     : round(fzm_score, 4),
                        "is_new"          : fzm_new,
                        "id_status"       : fzm_status,
                        "top3_labels"     : " | ".join(fzm_top3),
                        "bbox"            : (mx1, my1, mx2, my2),
                    })

                    if show_summary:
                        print(f"    {fzm_id:<12} {fzm_tinfo['display']:<22} "
                              f"Partial_Body   {fzm_vp:<18} "
                              f"match={fzm_score:.2f}  {fzm_status}  [FenceMask_OIV7]")
                    break   # one mask-search tiger per fence detection

            if valid_count == 1:
                if show_summary:
                    print(f"    [FENCE-MASK] Mask search also found no new tiger.")

                if _fz_tiger_signal_found:
                    # YOLO found tiger-class evidence in the fence zone but all
                    # guards failed — the model sees SOMETHING but cannot confirm
                    # it cleanly. Flag for human review.
                    result["needs_confirmation"] = True
                    _fence_reason = (
                        "fence/barrier present AND model detected a low-confidence "
                        "tiger-class signal in the fence zone that could not be "
                        "confirmed — possible second tiger occluded behind fence bars. "
                        "Human count required.")
                    result.setdefault("uncertain_detections", [])
                    for _fd in result["detections"]:
                        result["uncertain_detections"].append(
                            (_fd["tiger_id"], _fence_reason))
                    if show_summary:
                        print(f"\n    {'=' * 58}")
                        print(f"    ⚠  FENCE — UNRESOLVED SIGNAL — HUMAN REVIEW NEEDED")
                        print(f"    {'=' * 58}")
                        print(f"    MY PREDICTION : {valid_count} tiger(s) detected confidently")
                        print(f"    REASON        : Fence present + low-conf tiger signal")
                        print(f"                   found but could not be confirmed.")
                        print(f"    PLEASE CONFIRM: Is this the correct tiger count?")
                        print(f"    {'=' * 58}")
                else:
                    # ── FIX D revised (Bug Fix 18): DO NOT blanket-flag fence ─
                    # Original Fix D flagged EVERY fence+1-tiger image for human
                    # review even when all scans found zero evidence. This created
                    # massive noise (6-8 false flags per dataset) while burying
                    # real review-worthy images. Revised: if all automated scans
                    # (FenceZone + FenceMask at conf=0.01) found absolutely NO
                    # tiger signal behind the fence, trust the result silently.
                    # Only flag when there IS an unresolved signal (handled by
                    # the branch above). Fence presence alone ≠ count uncertainty.
                    if show_summary:
                        print(f"    [FENCE-ZONE] No tiger signal behind fence — no flag raised.")

    # ── CONFIDENCE ASSESSMENT (per detection) ────────────────────────────────────
    # RULE: Flag UNCERTAIN only when the pipeline has NO trained verification
    # mechanism for the detection. Everything that passed the species gate
    # (ResNet50 top-3 confirmation) is considered verified.
    #
    # CONFIDENCE LEVELS:
    #   HIGH   — OIV7 direct hit + strong re-capture match (>= 0.83).
    #   MEDIUM — Any detection that passed the species gate: Remainder scan,
    #             Margin scan, Whole_Image_Fallback, COCO_Proxy, LowConf_OIV7.
    #             These all went through verify_species_top3() + similarity
    #             guards — they are trained, calibrated features.
    #   LOW    — ONLY the following genuinely unverified cases:
    #             • Grid_Scan splits: halves are verified individually but the
    #               SPLIT DECISION itself (same tiger vs two tigers) has no
    #               species gate — it relies solely on fingerprint similarity.
    #             • New enrollment with match score < 0.45: barely any DB match
    #               at all — the pipeline is genuinely guessing.
    #
    # WHAT IS NOT FLAGGED (trained patterns, do not re-ask):
    #   • Remainder_L/R/T/B  — species gate + similarity + cross-half guard ✓
    #   • Margin_L/R/T/B     — species gate + similarity guard ✓
    #   • Whole_Image_Fallback — species gate passed, tiger confirmed ✓
    #   • LowConf_OIV7       — species gate passed, similarity guard ✓
    #   • COCO_Proxy         — species gate passed ✓
    #   • New enrollment 0.45–0.83 — normal range for first sighting of a tiger

    NEW_ENROLL_CONF_MIN = 0.45   # only flag if match is extremely low (no DB match at all)

    uncertain_detections = []
    for det in result["detections"]:
        src    = det.get("detection_source", "")
        score  = det.get("match_score", 0.0)
        is_new = det.get("is_new", False)
        reason = None

        # ── Bug Fix 18 cont'd: Flag detections that genuinely need review ──
        # 1. Grid scan splits (no species gate on split decision)
        # 2. New enrollments with very low match (no DB match at all)
        # 3. Tiny corner traces (tiger barely in frame — may be partial/missed)
        vis = det.get("visibility", "")
        bbox = det.get("bbox", (0, 0, 0, 0))
        _bw = max(1, bbox[2] - bbox[0])
        _bh = max(1, bbox[3] - bbox[1])
        _box_area_pct = (_bw * _bh) / max(1, img_area) * 100

        if src.startswith("Grid_Scan"):
            reason = (f"grid-split detection — split decision uses fingerprint "
                      f"similarity only, no species gate on the split itself")
        elif is_new and score < NEW_ENROLL_CONF_MIN:
            reason = (f"new enrollment with extremely low match score {score:.2f} "
                      f"— no close DB match found at all")
        elif vis == "Corner_Trace" and _box_area_pct < 8:
            reason = (f"tiny corner trace ({_box_area_pct:.1f}% of frame) — "
                      f"tiger barely visible at edge, count may be incomplete")

        if reason:
            det["confidence"] = "LOW"
            uncertain_detections.append((det["tiger_id"], reason))
        elif score >= 0.83 and not is_new and src == "OIV7_Direct":
            det["confidence"] = "HIGH"
        else:
            det["confidence"] = "MEDIUM"

    # Merge with any pre-set flags (e.g. fence-occlusion flag set earlier)
    prior_confirmation = result.get("needs_confirmation", False)
    prior_uncertain    = result.get("uncertain_detections", [])
    # Preserve entries from prior stages that aren't duplicated by this assessment
    prior_only = [e for e in prior_uncertain
                  if e not in uncertain_detections]
    result["uncertain_detections"] = prior_only + uncertain_detections
    result["needs_confirmation"]   = prior_confirmation or len(uncertain_detections) > 0

    if uncertain_detections and show_summary:
        unique_ids = list(dict.fromkeys(d["tiger_id"] for d in result["detections"]))
        unique_count = len(unique_ids)
        print(f"\n    {'=' * 58}")
        print(f"    ⚠  UNCERTAIN — HUMAN CONFIRMATION NEEDED")
        print(f"    {'=' * 58}")
        print(f"    MY PREDICTION : {unique_count} unique tiger(s) in this image")
        print(f"    PREDICTED IDs : {', '.join(unique_ids)}")
        print(f"    UNCERTAIN DETECTIONS:")
        for tid, why in uncertain_detections:
            print(f"      • {tid} — {why}")
        print(f"    PLEASE CONFIRM: is the tiger count correct?")
        print(f"    {'=' * 58}")

    # ── ZERO-DETECTION HUMAN REVIEW FLAG ────────────────────────────────────
    # If the pipeline found zero tigers, the species gate may have mis-classified
    # a real tiger (dark/blurry/partial images where ResNet50 top-3 returns non-tiger
    # classes). Rather than silently discarding these images, flag them for a human
    # to verify — the model might be wrong.
    # Rule: only flag when the image passed triage (not skipped for quality) but the
    # species gate rejected it. Genuinely empty forest images won't have any YOLO
    # boxes, so we check whether cascade detection DID find a box.
    if valid_count == 0 and not result.get("skipped", False):
        _reject_reason = (
            f"No tiger confirmed by species gate — model returned 0 detections. "
            f"Please verify manually: the species gate may have mis-classified a "
            f"real tiger in a dark, partial, or unusual pose image.")
        result["needs_confirmation"] = True
        result.setdefault("uncertain_detections", [])
        result["uncertain_detections"].append(("No_Detection", _reject_reason))
        if show_summary:
            print(f"\n    {'=' * 58}")
            print(f"    ⚠  ZERO DETECTIONS — HUMAN VERIFICATION NEEDED")
            print(f"    {'=' * 58}")
            print(f"    The species gate returned 0 tigers for this image.")
            print(f"    Please check the image manually — model may be wrong.")
            print(f"    {'=' * 58}")

    # ── FIX E: Count UNIQUE tiger IDs, not total detections (Bug Fix 2026-06-12)
    # Ground truth: 000903.jpg has 1 tiger but 2 YOLO boxes both map to Tiger_001.
    # Previous code: tiger_count = valid_count (total detections = 2). WRONG.
    # New code: tiger_count = number of UNIQUE tiger IDs in detections list.
    # valid_count still tracks total detections (used by scan gates).
    unique_tiger_ids = set(d["tiger_id"] for d in result["detections"])
    result["tiger_count"] = len(unique_tiger_ids)
    if result["tiger_count"] > 0:
        cv2.imwrite(str(ANNOTATED_DIR / img_path.name), annotated)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 12b — HTML DASHBOARD GENERATOR
# Produces an interactive self-contained HTML report after every batch run.
# No external server needed — open the .html file directly in any browser.
# ══════════════════════════════════════════════════════════════════════════════

def _generate_html_dashboard(all_results, csv_rows, conservative_count, liberal_count,
                              source_counts, morph_counts, total_dets, unusable_count,
                              no_tiger_count, tiger_img_count, html_path, ts, images):
    """Build a self-contained HTML dashboard from batch results."""

    # ── Per-image rows ────────────────────────────────────────────────────────
    img_rows_html = ""
    for res in all_results:
        if res["skipped"]:
            status_badge = '<span class="badge badge-unusable">Unusable</span>'
            count_cell   = "—"
            ids_cell     = "—"
            review_cell  = "—"
        elif res["tiger_count"] == 0:
            status_badge = '<span class="badge badge-no">No Tiger</span>'
            count_cell   = "0"
            ids_cell     = "—"
            review_cell  = "—"
        else:
            status_badge = '<span class="badge badge-yes">Tiger ✓</span>'
            count_cell   = str(res["tiger_count"])
            ids_cell     = ", ".join(sorted({d["tiger_id"] for d in res["detections"]}))
            needs_review = res.get("needs_confirmation", False)
            review_cell  = ('<span class="badge badge-review">⚠ REVIEW</span>'
                            if needs_review else
                            '<span class="badge badge-ok">OK</span>')

        img_rows_html += f"""
        <tr>
          <td><code>{res['filename']}</code></td>
          <td>{status_badge}</td>
          <td>{count_cell}</td>
          <td>{ids_cell}</td>
          <td>{review_cell}</td>
          <td><small>{res.get('quality_bucket','—')}</small></td>
        </tr>"""

    # ── Detection detail rows ─────────────────────────────────────────────────
    det_rows_html = ""
    for row in csv_rows:
        conf   = row.get("Confidence", "MEDIUM")
        review = row.get("Human_Review", "NO")
        conf_badge = (f'<span class="badge badge-high">HIGH</span>'   if conf == "HIGH"   else
                      f'<span class="badge badge-medium">MEDIUM</span>' if conf == "MEDIUM" else
                      f'<span class="badge badge-review">LOW ⚠</span>')
        rev_badge  = ('<span class="badge badge-review">YES ⚠</span>'
                      if review == "YES" else
                      '<span class="badge badge-ok">NO</span>')
        det_rows_html += f"""
        <tr>
          <td><code>{row['Image_File']}</code></td>
          <td>{row['Tiger_ID']}</td>
          <td>{row.get('Is_New_Tiger','')}</td>
          <td>{row.get('ViewPoint','')}</td>
          <td>{row.get('Detection_Source','')}</td>
          <td>{row.get('Match_Score','')}</td>
          <td>{conf_badge}</td>
          <td>{rev_badge}</td>
          <td><small>{row.get('Color_Morph','')}</small></td>
        </tr>"""

    # ── Human Review Required section — ALWAYS shown ─────────────────────────
    # Rule: Every image the model is not fully confident about must appear here.
    # Categories:
    #   1. needs_confirmation=True  → fence occlusion, vertical body, low match
    #   2. skipped=True             → rejected by species gate / unusable
    #      (rejected images may be genuine tigers that ResNet50 mis-classified;
    #       a human should verify before discarding them)
    review_images = [r for r in all_results
                     if r.get("needs_confirmation") or r.get("skipped")]

    items = ""
    for res in review_images:
        if res.get("skipped"):
            tag    = "REJECTED / UNUSABLE"
            color  = "#721C24"
            bg     = "#F8D7DA"
            reason = (f"Image was rejected by the species gate or marked unusable. "
                      f"Quality: {res.get('quality_bucket','Unknown')}. "
                      f"Please verify manually — the model may have incorrectly "
                      f"dismissed a real tiger.")
            pred   = "0 (rejected)"
        else:
            pred_ids = ", ".join(dict.fromkeys(d["tiger_id"] for d in res["detections"]))
            pred   = f"{res['tiger_count']} tiger(s): {pred_ids}"
            reason_parts = []
            for tid, why in res.get("uncertain_detections", []):
                reason_parts.append(f"• [{tid}] {why}")
            reason = "<br>".join(reason_parts) if reason_parts else "Low confidence detection."
            # Choose tag based on primary reason
            if any("fence" in w.lower() for _, w in res.get("uncertain_detections", [])):
                tag = "FENCE OCCLUSION"
                color = "#856404"; bg = "#FFF3CD"
            elif any("vertical" in w.lower() for _, w in res.get("uncertain_detections", [])):
                tag = "VERTICAL BODY"
                color = "#0C5460"; bg = "#D1ECF1"
            else:
                tag = "LOW CONFIDENCE"
                color = "#383D41"; bg = "#E2E3E5"

        items += f"""
        <div class="uncertain-card" style="border-left:4px solid {color};background:{bg};
             padding:10px 14px;margin-bottom:10px;border-radius:4px;">
          <strong style="color:{color}">[{tag}]</strong>
          &nbsp;<strong>{res['filename']}</strong>
          &nbsp;— My prediction: <em>{pred}</em><br>
          <span class="reason" style="font-size:0.88em">{reason}</span>
        </div>"""

    if not review_images:
        items = """<div style="padding:10px;color:#155724;background:#D4EDDA;
                   border-radius:4px;">
                   ✓ All detections in this batch are HIGH or MEDIUM confidence.
                   No human review required.</div>"""

    uncertain_html = f"""
    <div class="section">
      <h2>⚠ Human Review Required — {len(review_images)} image(s)</h2>
      <p style="font-size:0.9em;color:#555;">
        The images below need a human to verify the count or confirm the tiger's presence.
        This includes: fence-occluded tigers, low-confidence detections, vertical-body images
        where the multi-tiger scan was skipped, and images rejected by the species gate.
      </p>
      {items}
    </div>"""

    # ── Source breakdown bars ─────────────────────────────────────────────────
    src_bars = ""
    for src, cnt in sorted(source_counts.items(), key=lambda x: -x[1]):
        pct = round(cnt / total_dets * 100, 1) if total_dets else 0
        src_bars += f"""
        <div class="bar-row">
          <span class="bar-label">{src}</span>
          <div class="bar-track"><div class="bar-fill" style="width:{pct}%"></div></div>
          <span class="bar-val">{cnt} ({pct}%)</span>
        </div>"""

    # ── Morph breakdown bars ──────────────────────────────────────────────────
    morph_bars = ""
    for morph, cnt in sorted(morph_counts.items(), key=lambda x: -x[1]):
        pct  = round(cnt / total_dets * 100, 1) if total_dets else 0
        disp = TIGER_TYPES.get(morph, {}).get("display", morph)
        morph_bars += f"""
        <div class="bar-row">
          <span class="bar-label">{disp}</span>
          <div class="bar-track"><div class="bar-fill bar-morph" style="width:{pct}%"></div></div>
          <span class="bar-val">{cnt} ({pct}%)</span>
        </div>"""

    human_review_count = sum(1 for r in csv_rows if r.get("Human_Review") == "YES")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TRACE Tiger Pipeline — V2 Dashboard</title>
<style>
  :root {{
    --primary: #0D2B55;
    --accent:  #1A4F8A;
    --light:   #E3F2FD;
    --warn:    #FF6B35;
    --ok:      #2ECC71;
    --med:     #F39C12;
    --low:     #E74C3C;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #F4F6F9; color: #222; }}
  header {{ background: var(--primary); color: #fff; padding: 24px 32px; }}
  header h1 {{ font-size: 1.6rem; letter-spacing: 1px; }}
  header p  {{ font-size: 0.85rem; opacity: 0.75; margin-top: 4px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
           gap: 16px; padding: 24px 32px 0; }}
  .card {{ background: #fff; border-radius: 8px; padding: 20px; text-align: center;
           box-shadow: 0 2px 6px rgba(0,0,0,.08); }}
  .card .num {{ font-size: 2.2rem; font-weight: 700; color: var(--primary); }}
  .card .lbl {{ font-size: 0.78rem; color: #666; margin-top: 4px; }}
  .card.warn  .num {{ color: var(--warn); }}
  .card.ok    .num {{ color: var(--ok);   }}
  .section {{ background: #fff; border-radius: 8px; margin: 20px 32px;
              padding: 20px 24px; box-shadow: 0 2px 6px rgba(0,0,0,.08); }}
  .section h2 {{ font-size: 1rem; color: var(--primary); margin-bottom: 14px;
                 border-bottom: 2px solid var(--light); padding-bottom: 8px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.84rem; }}
  th {{ background: var(--primary); color: #fff; padding: 8px 10px; text-align: left; }}
  td {{ padding: 7px 10px; border-bottom: 1px solid #eee; }}
  tr:hover td {{ background: var(--light); }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 12px;
            font-size: 0.75rem; font-weight: 600; }}
  .badge-yes      {{ background: #D4EDDA; color: #155724; }}
  .badge-no       {{ background: #F8F9FA; color: #6C757D; }}
  .badge-unusable {{ background: #F8D7DA; color: #721C24; }}
  .badge-review   {{ background: #FFF3CD; color: #856404; }}
  .badge-ok       {{ background: #D4EDDA; color: #155724; }}
  .badge-high     {{ background: #CCE5FF; color: #004085; }}
  .badge-medium   {{ background: #D4EDDA; color: #155724; }}
  .bar-row  {{ display: flex; align-items: center; margin-bottom: 8px; }}
  .bar-label{{ width: 220px; font-size: 0.82rem; color: #444; }}
  .bar-track{{ flex: 1; background: #EEE; border-radius: 4px; height: 14px; }}
  .bar-fill {{ background: var(--accent); height: 14px; border-radius: 4px;
               min-width: 2px; }}
  .bar-morph{{ background: var(--warn); }}
  .bar-val  {{ width: 100px; text-align: right; font-size: 0.8rem; color: #666; }}
  .uncertain-card {{ background: #FFF8E1; border-left: 4px solid var(--warn);
                     border-radius: 4px; padding: 12px; margin-bottom: 10px; }}
  .reason   {{ color: #6C3F00; font-size: 0.82rem; margin-top: 4px; }}
  footer {{ text-align: center; padding: 20px; font-size: 0.78rem; color: #999; }}
</style>
</head>
<body>
<header>
  <h1>TRACE Tiger Pipeline — V2 Dashboard</h1>
  <p>EPAIB Batch 05 | Group 4 | IIM Lucknow &nbsp;|&nbsp; Generated: {ts} &nbsp;|&nbsp;
     Survey: {RANGE_NAME} | {BEAT_NAME}</p>
</header>

<div class="grid">
  <div class="card"><div class="num">{len(images)}</div><div class="lbl">Total Images</div></div>
  <div class="card ok"><div class="num">{tiger_img_count}</div><div class="lbl">Images with Tigers</div></div>
  <div class="card"><div class="num">{no_tiger_count}</div><div class="lbl">No Tiger</div></div>
  <div class="card"><div class="num">{unusable_count}</div><div class="lbl">Unusable / Blurry</div></div>
  <div class="card"><div class="num">{total_dets}</div><div class="lbl">Total Detections</div></div>
  <div class="card"><div class="num">{conservative_count}</div><div class="lbl">Conservative Count</div></div>
  <div class="card"><div class="num">{liberal_count}</div><div class="lbl">Liberal Count</div></div>
  <div class="card {'warn' if human_review_count > 0 else 'ok'}">
    <div class="num">{human_review_count}</div>
    <div class="lbl">Need Human Review</div>
  </div>
</div>

{uncertain_html}

<div class="section">
  <h2>Per-Image Summary</h2>
  <table>
    <thead><tr>
      <th>Image</th><th>Status</th><th>Count</th>
      <th>Tiger IDs</th><th>Review</th><th>Quality</th>
    </tr></thead>
    <tbody>{img_rows_html}</tbody>
  </table>
</div>

<div class="section">
  <h2>Detection Detail (one row per detection)</h2>
  <table>
    <thead><tr>
      <th>Image</th><th>Tiger ID</th><th>New?</th>
      <th>ViewPoint</th><th>Source</th><th>Match Score</th>
      <th>Confidence</th><th>Human Review</th><th>Morph</th>
    </tr></thead>
    <tbody>{det_rows_html}</tbody>
  </table>
</div>

<div class="section">
  <h2>Detection Source Breakdown</h2>
  {src_bars}
</div>

<div class="section">
  <h2>Colour Morph Breakdown</h2>
  {morph_bars}
</div>

<footer>
  TRACE Tiger Pipeline V2 &nbsp;|&nbsp; EPAIB Batch 05 Group 4 &nbsp;|&nbsp; IIM Lucknow
</footer>
</body>
</html>"""

    html_path.write_text(html, encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 13 — BATCH PROCESSING + VIEWPOINT-AWARE POPULATION REPORT
# Dr. Bose's viewpoint census + TRACE's complete report format
# ══════════════════════════════════════════════════════════════════════════════

def run_population_report(data_dir=None, run_gradcam=False):
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    images   = sorted([p for p in data_dir.rglob("*")
                       if p.suffix.lower() in SUPPORTED_EXTS])
    if not images:
        print(f"[ERROR] No images found in {data_dir}")
        return

    print("=" * 70)
    print("  TIGER AI ULTIMATE — POPULATION REPORT  (v3.0 Fused Pipeline)")
    print(f"  Survey : {RANGE_NAME} | {BEAT_NAME}")
    print(f"  Images : {len(images)} | Data : {data_dir}")
    print("=" * 70)

    load_db()

    all_results     = []
    csv_rows        = []
    unusable_count  = 0
    no_tiger_count  = 0
    tiger_img_count = 0
    total_dets      = 0

    for idx, img_path in enumerate(images, 1):
        print(f"\n[{idx:3d}/{len(images)}]", end=" ")
        res = analyze_image(img_path, show_summary=True, run_gradcam=run_gradcam)
        all_results.append(res)

        if res["skipped"]:
            unusable_count += 1
        elif res["tiger_count"] == 0:
            no_tiger_count += 1
        else:
            tiger_img_count += 1
            total_dets      += res["tiger_count"]

        for d in res["detections"]:
            csv_rows.append({
                "Image_File"       : res["filename"],
                "Range"            : RANGE_NAME,
                "Beat"             : BEAT_NAME,
                "Image_Quality"    : res["quality_bucket"],
                "Has_Tiger"        : "Yes" if res["tiger_count"] > 0 else "No",
                "Tigers_In_Frame"  : res["tiger_count"],
                "Tiger_ID"         : d["tiger_id"],
                "Is_New_Tiger"     : d["is_new"],
                "Visibility"       : d["visibility"],
                "ViewPoint"        : d["viewpoint"],
                "Detection_Source" : d["detection_source"],
                "Color_Morph"      : d["morph"],
                "ID_Status"        : d["id_status"],
                "Match_Score"      : d["match_score"],
                "Detection_Conf"   : d["det_conf"],
                "Top3_ResNet"      : d["top3_labels"],
                "Confidence"       : d.get("confidence", "MEDIUM"),
                "Human_Review"     : "YES" if d.get("confidence") == "LOW" else "NO",
            })

    if csv_rows:
        pd.DataFrame(csv_rows).to_csv(OUTPUT_CSV, index=False)

    # ── Per-image summary CSV (one row per image, not per detection) ──────
    image_summary_rows = []
    for res in all_results:
        tiger_ids = ", ".join(sorted({d["tiger_id"] for d in res["detections"]})) \
                    if res["detections"] else "—"
        morphs    = ", ".join(sorted({d["display"] for d in res["detections"]})) \
                    if res["detections"] else "—"
        image_summary_rows.append({
            "Image_File"    : res["filename"],
            "Image_Quality" : res["quality_bucket"],
            "Has_Tiger"     : "Yes" if res["tiger_count"] > 0 else
                              ("Unusable" if res["skipped"] else "No"),
            "Tiger_Count"   : res["tiger_count"],
            "Tiger_IDs"     : tiger_ids,
            "Color_Morphs"  : morphs,
        })

    img_summary_csv = REPORTS_DIR / "ultimate_image_summary.csv"
    pd.DataFrame(image_summary_rows).to_csv(img_summary_csv, index=False)

    save_db()

    # ── ViewPoint-aware census (from Dr. Bose) ────────────────────────────
    # For each unique tiger ID, collect all viewpoints it was seen from.
    # Conservative count = max(L, R, F) — avoids counting L+R as 2 tigers.
    viewpoint_map = {}
    for row in csv_rows:
        tid = row["Tiger_ID"]
        vp  = row["ViewPoint"]
        if tid not in viewpoint_map:
            viewpoint_map[tid] = set()
        viewpoint_map[tid].add(vp)

    left_tigers   = {tid for tid, vps in viewpoint_map.items() if "Left_Flank"  in vps}
    right_tigers  = {tid for tid, vps in viewpoint_map.items() if "Right_Flank" in vps}
    front_tigers  = {tid for tid, vps in viewpoint_map.items() if "Frontal"     in vps}
    all_tigers    = left_tigers | right_tigers | front_tigers

    conservative_count = max(len(left_tigers), len(right_tigers), len(front_tigers))
    liberal_count      = len(all_tigers)

    # ── Detection source breakdown ────────────────────────────────────────
    source_counts = {}
    for row in csv_rows:
        src = row["Detection_Source"]
        source_counts[src] = source_counts.get(src, 0) + 1

    # ── Morph breakdown ───────────────────────────────────────────────────
    morph_counts = {}
    for row in csv_rows:
        m = row["Color_Morph"]
        morph_counts[m] = morph_counts.get(m, 0) + 1

    # ── Build report ──────────────────────────────────────────────────────
    ts    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "=" * 65,
        "   FOREST TIGER POPULATION REPORT — v3.0 FUSED PIPELINE",
        "=" * 65,
        f"  Generated        : {ts}",
        f"  Survey Location  : {RANGE_NAME} | {BEAT_NAME}",
        f"  Detection Chain  : YOLOv8l-OIV7 -> COCO-proxy -> Whole-image  [V2]",
        f"  Species Gate     : EfficientNetV2-L top-3 tiger verification [V2]",
        f"  Identity Method  : Cosine Similarity (threshold {SIMILARITY_THRESHOLD})",
        f"  ViewPoint Census : Left_Flank / Right_Flank / Frontal",
        "=" * 65,
        "",
        "  IMAGE PROCESSING SUMMARY",
        "  " + "-" * 40,
        f"  Total images processed  : {len(images)}",
        f"  Images with tigers      : {tiger_img_count}",
        f"  Images with no tiger    : {no_tiger_count}",
        f"  Unusable images         : {unusable_count}",
        f"  Total tiger detections  : {total_dets}",
        "",
        "  PER-IMAGE BREAKDOWN",
        "  " + "-" * 40,
        f"  {'Image':<32} {'Quality':<22} {'Tiger?':<10} {'Count':<7} Tiger IDs",
        "  " + "-" * 95,
    ]

    for res in all_results:
        has_t  = "Unusable" if res["skipped"] else ("YES" if res["tiger_count"] > 0 else "no")
        ids    = ", ".join(sorted({d["tiger_id"] for d in res["detections"]})) \
                 if res["detections"] else "—"
        count  = str(res["tiger_count"]) if not res["skipped"] else "—"
        lines.append(
            f"  {res['filename']:<32} {res['quality_bucket']:<22} {has_t:<10} {count:<7} {ids}")

    lines += [
        "  " + "-" * 95,
        "",
        "  CENSUS RESULTS",
        "  " + "-" * 40,
        f"  Raw unique IDs in DB    : {len(tiger_db)}",
        f"  ViewPoint-aware count",
        f"    Left-flank only       : {len(left_tigers)} unique IDs",
        f"    Right-flank only      : {len(right_tigers)} unique IDs",
        f"    Frontal only          : {len(front_tigers)} unique IDs",
        f"  CONSERVATIVE ESTIMATE   : {conservative_count}",
        f"  (= max of L/R/F counts — avoids counting same tiger twice)",
        f"  LIBERAL ESTIMATE        : {liberal_count}",
        f"  (= union of all IDs — upper bound)",
        "",
        "  DETECTION SOURCE BREAKDOWN",
        "  " + "-" * 40,
    ]

    for src, cnt in sorted(source_counts.items(), key=lambda x: -x[1]):
        pct  = cnt / total_dets * 100 if total_dets else 0
        bar  = "#" * int(pct / 5)
        lines.append(f"  {src:<28}: {cnt:3d}  ({pct:5.1f}%)  {bar}")

    lines += [
        "",
        "  COLOUR MORPH BREAKDOWN",
        "  " + "-" * 40,
    ]

    for morph, cnt in sorted(morph_counts.items(), key=lambda x: -x[1]):
        pct  = cnt / total_dets * 100 if total_dets else 0
        bar  = "#" * int(pct / 4)
        disp = TIGER_TYPES.get(morph, {}).get("display", morph)
        lines.append(f"  {disp:<26}: {cnt:3d}  ({pct:5.1f}%)  {bar}")

    lines += [
        "",
        "  INDIVIDUAL SIGHTING COUNTS",
        "  " + "-" * 40,
    ]
    for tid, cnt in sorted(tiger_counts.items()):
        vps   = ", ".join(sorted(viewpoint_map.get(tid, {"Unknown"})))
        lines.append(f"  {tid:<14}: {cnt} sighting(s)   ViewPoints: {vps}")

    # ── Human Review section — ALWAYS shown ──────────────────────────────────
    # Includes: low-confidence flags, fence occlusion, vertical-body guard,
    # AND rejected/skipped images (species gate may occasionally mis-fire).
    review_images_txt = [
        res for res in all_results
        if res.get("needs_confirmation") or res.get("skipped")
    ]

    lines += [
        "",
        "  " + "=" * 61,
        f"  ⚠  HUMAN REVIEW REQUIRED — {len(review_images_txt)} IMAGE(S)",
        "  " + "=" * 61,
        "  Every image listed below needs a human to verify the count",
        "  or confirm the tiger's presence before results are finalised.",
        "  Categories:",
        "    [FENCE]    — fence/barrier detected; possible 2nd tiger hidden",
        "    [VERTICAL] — vertical body orientation; multi-tiger scan skipped",
        "    [LOW CONF] — match score too low to trust auto-identification",
        "    [REJECTED] — species gate rejected; verify not a missed tiger",
        "  " + "-" * 61,
    ]

    if not review_images_txt:
        lines.append(
            "  ✓ All detections HIGH/MEDIUM confidence. No review needed.")
    else:
        for res in review_images_txt:
            if res.get("skipped"):
                lines.append(f"  [REJECTED]  {res['filename']}")
                lines.append(f"    My prediction : 0 tigers (rejected by species gate)")
                lines.append(f"    Quality       : {res.get('quality_bucket','Unknown')}")
                lines.append(f"    Action needed : Verify manually — model may have")
                lines.append(f"                   incorrectly dismissed a real tiger.")
            else:
                pred_ids = ", ".join(
                    dict.fromkeys(d["tiger_id"] for d in res["detections"]))
                reasons  = res.get("uncertain_detections", [])
                # Determine primary flag
                if any("fence" in w.lower() for _, w in reasons):
                    flag = "[FENCE]   "
                elif any("vertical" in w.lower() for _, w in reasons):
                    flag = "[VERTICAL]"
                else:
                    flag = "[LOW CONF]"
                lines.append(f"  {flag}  {res['filename']}")
                lines.append(
                    f"    My prediction : {res['tiger_count']} tiger(s)  →  {pred_ids}")
                for tid, why in reasons:
                    lines.append(f"    Reason ({tid:<14}): {why[:80]}")
            lines.append("")

    lines += ["  " + "=" * 61]

    lines += [
        "",
        "  OUTPUT FILES",
        "  " + "-" * 40,
        f"  CSV Report       : {OUTPUT_CSV}",
        f"  Image Summary    : {img_summary_csv}",
        f"  Identity DB      : {DB_FILE}",
        f"  Annotated images : {ANNOTATED_DIR}",
        f"  Grad-CAM outputs : {GRADCAM_DIR}",
        "",
        "=" * 65,
        "  END OF REPORT — TIGER AI ULTIMATE v3.0",
        "=" * 65,
    ]

    report = "\n".join(lines)
    print("\n\n" + report)
    REPORT_TXT.write_text(report, encoding="utf-8")
    print(f"\n[INFO] Report saved -> {REPORT_TXT}")
    print(f"[INFO] CSV    saved -> {OUTPUT_CSV}")
    print(f"[INFO] DB     saved -> {DB_FILE}")

    # ── HTML Dashboard ────────────────────────────────────────────────────────
    html_path = REPORTS_DIR / "v2_dashboard.html"
    _generate_html_dashboard(
        all_results=all_results,
        csv_rows=csv_rows,
        conservative_count=conservative_count,
        liberal_count=liberal_count,
        source_counts=source_counts,
        morph_counts=morph_counts,
        total_dets=total_dets,
        unusable_count=unusable_count,
        no_tiger_count=no_tiger_count,
        tiger_img_count=tiger_img_count,
        html_path=html_path,
        ts=ts,
        images=images,
    )
    print(f"[INFO] HTML dashboard -> {html_path}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tiger AI Ultimate — End-to-End Pipeline v3.0  |  EPAIB Batch 05 Group 4",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
INFERENCE (default):
  py -3 src/tiger_ai_ultimate.py                         # batch run on sample data
  py -3 src/tiger_ai_ultimate.py --data path/to/images   # custom image folder
  py -3 src/tiger_ai_ultimate.py --image tiger.jpg       # single image
  py -3 src/tiger_ai_ultimate.py --gradcam               # with Grad-CAM heatmaps

TRAINING:
  py -3 src/tiger_ai_ultimate.py --train --stage yolo --data-yaml data/tiger_dataset.yaml
  py -3 src/tiger_ai_ultimate.py --train --stage detection --train-data data/processed
  py -3 src/tiger_ai_ultimate.py --train --stage identification --train-data data/processed
        """,
    )
    # ── Inference flags ───────────────────────────────────────────────────────
    parser.add_argument("--image",   type=str, default=None,
                        help="Analyse a single image by filename or full path")
    parser.add_argument("--data",    type=str, default=None,
                        help="Path to image folder for batch inference")
    parser.add_argument("--gradcam", action="store_true",
                        help="Generate Grad-CAM visualisations for each detection")
    # ── Training flags ────────────────────────────────────────────────────────
    parser.add_argument("--train",   action="store_true",
                        help="Run training instead of inference")
    parser.add_argument("--stage",   type=str, default="yolo",
                        choices=["yolo", "detection", "identification"],
                        help="Which training stage to run (default: yolo)")
    parser.add_argument("--train-data", type=str, default="data/processed",
                        help="Path to processed training data folder")
    parser.add_argument("--data-yaml",  type=str, default="data/tiger_dataset.yaml",
                        help="Path to YOLO data.yaml (yolo stage only)")
    parser.add_argument("--epochs",     type=int, default=None,
                        help="Override number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Training batch size (default 16; reduce to 8 on CPU)")
    parser.add_argument("--model-dir",  type=str, default="models",
                        help="Folder to save trained model weights")
    parser.add_argument("--num-tigers", type=int, default=50,
                        help="Number of individual tiger classes (identification stage)")
    args = parser.parse_args()

    if args.train:
        # ── TRAINING MODE ─────────────────────────────────────────────────────
        print("\n" + "█" * 65)
        print("  TIGER AI ULTIMATE — TRAINING MODE")
        print("  EPAIB Batch 05 | Group 4 | IIM Lucknow")
        print("█" * 65)
        if args.stage == "yolo":
            epochs = args.epochs or 50
            train_yolo(
                data_yaml=args.data_yaml,
                epochs=epochs,
                batch=args.batch_size,
                model_dir=args.model_dir,
            )
        elif args.stage == "detection":
            epochs = args.epochs or 30
            train_detection_model(
                data_dir=args.train_data,
                model_dir=args.model_dir,
                epochs_phase1=max(1, epochs - 10),
                epochs_phase2=min(10, epochs),
                batch_size=args.batch_size,
            )
        elif args.stage == "identification":
            epochs = args.epochs or 30
            train_identification_model(
                data_dir=args.train_data,
                model_dir=args.model_dir,
                num_tigers=args.num_tigers,
                epochs=epochs,
                batch_size=args.batch_size,
            )
    else:
        # ── INFERENCE MODE ────────────────────────────────────────────────────
        if args.image:
            load_db()
            img_path = DATA_DIR / args.image
            if not img_path.exists():
                img_path = Path(args.image)
            result = analyze_image(img_path, show_summary=True,
                                   run_gradcam=args.gradcam)
            save_db()
        else:
            run_population_report(data_dir=args.data, run_gradcam=args.gradcam)
