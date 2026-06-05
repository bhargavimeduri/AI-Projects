"""
Generates TRACE_Tiger_Pipeline.ipynb — ready to run on Google Colab.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    return nbf.v4.new_markdown_cell(text)

def code(text):
    return nbf.v4.new_code_cell(text)

# ══════════════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("""# TRACE — Tiger Recognition And Census Engine
## EPAIB Batch 05 | Group 4 | IIM Lucknow
**Real-world partner:** Wildlife Department, Government of West Bengal

---

### 3-Model Architecture
| Phase | Model | What It Does |
|-------|-------|-------------|
| Phase 1 | Image Triage | Reject unusable images before any ML runs |
| Phase 2 | YOLOv8 + ResNet50 | Detect tigers, count, identify individuals |
| Phase 3 | HSV Classifier | Classify color morph + subspecies |

**How to run:** Runtime → Run All  (or Shift+Enter cell by cell)
"""))

# ══════════════════════════════════════════════════════════════════════════════
# CELL 1 — Install dependencies
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("## Step 1: Install Dependencies"))
cells.append(code("""\
# Install all required packages
!pip install ultralytics -q
!pip install torch torchvision -q
!pip install opencv-python-headless -q
!pip install scikit-learn -q
!pip install pandas numpy Pillow -q
print("All packages installed.")
"""))

# ══════════════════════════════════════════════════════════════════════════════
# CELL 2 — Mount Google Drive
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("## Step 2: Mount Google Drive (Upload Your Tiger Images Here)"))
cells.append(code("""\
from google.colab import drive
drive.mount('/content/drive')

# ── Set this to your folder in Google Drive ────────────────────────────────
# Upload your tiger images to Google Drive first, then update this path.
# Example: if your images are in My Drive > tiger_images > train
IMAGE_DIR = "/content/drive/MyDrive/tiger_images/train"

# If you want to use sample images uploaded directly to Colab instead:
# IMAGE_DIR = "/content/tiger_images"

import os
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs("/content/reports", exist_ok=True)
print(f"Image folder: {IMAGE_DIR}")
"""))

# ══════════════════════════════════════════════════════════════════════════════
# CELL 3 — Imports & Config
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("## Step 3: Imports & Configuration"))
cells.append(code("""\
import json
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO
from pathlib import Path
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ── Field Metadata (edit before each survey session) ──────────────────────
RANGE_NAME = "Sundarbans Tiger Reserve"
BEAT_NAME  = "Block-A / Sector-3"

# ── Phase 1: Image Triage thresholds ──────────────────────────────────────
BLUR_THRESHOLD        = 80.0   # Laplacian variance below this = blurry
OVEREXPOSE_THRESHOLD  = 240    # Mean brightness above this = whiteout
UNDEREXPOSE_THRESHOLD = 20     # Mean brightness below this = blackout
IR_CHANNEL_DIFF       = 5      # Max channel diff = grayscale / IR night image

# ── Phase 2: Detection & Identity thresholds ──────────────────────────────
SIMILARITY_THRESHOLD    = 0.83  # Confirmed match — update DB
PARTIAL_MATCH_THRESHOLD = 0.70  # Possible match (half-body) — log only
CONFIDENCE_THRESHOLD    = 0.35  # YOLO minimum confidence
MIN_BOX_FULL            = 0.05  # >= 5% of frame = full body
MIN_BOX_PARTIAL         = 0.03  # 3-5% of frame = half body

# ── YOLO class filter ──────────────────────────────────────────────────────
# COCO has no tiger class. These are the closest visual proxies.
# After fine-tuning YOLO on tiger data, replace with {0}.
TIGER_PROXY_CLASSES = {15, 16, 17, 24}  # cat, dog, horse, zebra

# ── Paths ──────────────────────────────────────────────────────────────────
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
OUTPUT_CSV     = "/content/reports/tiger_enumeration_results.csv"
DB_FILE        = "/content/reports/tiger_registry_db.json"

# ── Device ─────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE.upper()}")
print(f"GPU available: {torch.cuda.is_available()}")
"""))

# ══════════════════════════════════════════════════════════════════════════════
# CELL 4 — Load Models
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("## Step 4: Load Models (YOLOv8 + ResNet50)"))
cells.append(code("""\
# ── YOLOv8 — Tiger Detection ───────────────────────────────────────────────
print("Loading YOLOv8 detector...")
detector = YOLO("yolov8n.pt")   # Downloads automatically on first run

# ── ResNet50 — Stripe Fingerprint Extractor ────────────────────────────────
# We remove the last 1000-class head ([:-1]) and keep the 2048-dim features.
# Those 2048 numbers ARE the stripe fingerprint for each tiger.
print("Loading ResNet50 feature extractor...")
resnet_base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
feature_extractor = torch.nn.Sequential(
    *(list(resnet_base.children())[:-1])
).to(DEVICE)
feature_extractor.eval()

# Standard ImageNet preprocessing: Resize(256) → CenterCrop(224)
# CenterCrop preserves aspect ratio — better than direct Resize(224,224)
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

print("Models loaded and ready.")
"""))

# ══════════════════════════════════════════════════════════════════════════════
# CELL 5 — Tiger Database
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("## Step 5: Tiger Database (Persists Across Runs)"))
cells.append(code("""\
tiger_database       = {}   # {"Tiger_001": np.array(2048,)}
tiger_sighting_counts = {}  # {"Tiger_001": int}
results_log          = []

def load_database():
    \"\"\"Load existing tiger registry from disk if available.\"\"\"
    if Path(DB_FILE).exists():
        try:
            with open(DB_FILE, "r") as f:
                raw = json.load(f)
            for tid, entry in raw.items():
                tiger_database[tid]        = np.array(entry["embedding"])
                tiger_sighting_counts[tid] = entry["sighting_count"]
            print(f"Loaded {len(tiger_database)} existing tiger profiles.")
        except Exception as e:
            print(f"Could not load database: {e} — starting fresh.")
    else:
        print("No existing database found — starting fresh.")

def save_database():
    \"\"\"Persist tiger registry to disk.\"\"\"
    serialisable = {
        tid: {
            "embedding":      tiger_database[tid].tolist(),
            "sighting_count": tiger_sighting_counts[tid]
        }
        for tid in tiger_database
    }
    with open(DB_FILE, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"Database saved: {len(tiger_database)} tigers.")

load_database()
"""))

# ══════════════════════════════════════════════════════════════════════════════
# CELL 6 — Phase 1: Image Triage
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
## Phase 1 — Image Triage
**Goal:** Reject unusable images BEFORE any ML model runs.

Every camera trap image is classified into one of these buckets:
| Bucket | Description |
|--------|------------|
| `Clean` | Good quality — pass to Phase 2 |
| `IR_Night` | Grayscale infrared night shot — still usable |
| `Unusable_Overexposed` | Whiteout (animal too close to IR flash) |
| `Unusable_Underexposed` | Blackout (too dark, flash didn't reach) |
| `Unusable_Blurry` | Motion blur (fast animal + slow shutter) |
"""))
cells.append(code("""\
def detect_ir_night(img_bgr):
    \"\"\"
    Camera traps use IR flash at night → grayscale images.
    When all 3 RGB channels are nearly identical, image is B&W.
    \"\"\"
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    diff_rg = np.std(rgb[:,:,0] - rgb[:,:,1])
    diff_rb = np.std(rgb[:,:,0] - rgb[:,:,2])
    return (diff_rg < IR_CHANNEL_DIFF) and (diff_rb < IR_CHANNEL_DIFF)

def detect_overexposure(img_bgr):
    \"\"\"Whiteout — animal too close to IR flash.\"\"\"
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray)) > OVEREXPOSE_THRESHOLD

def detect_underexposure(img_bgr):
    \"\"\"Blackout — too far from camera or dense canopy.\"\"\"
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray)) < UNDEREXPOSE_THRESHOLD

def detect_blur(img_bgr):
    \"\"\"
    Motion blur detection using Laplacian variance.
    Sharp image = strong edges = high variance.
    Blurry image = weak edges = low variance.
    \"\"\"
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var()) < BLUR_THRESHOLD

def triage_image(img_bgr):
    \"\"\"
    Full triage — returns (quality_bucket, note, should_skip).
    should_skip=True means image is unusable, don't run ML.
    \"\"\"
    if detect_overexposure(img_bgr):
        return "Unusable_Overexposed", "IR flash whiteout", True
    if detect_underexposure(img_bgr):
        return "Unusable_Underexposed", "Blackout image", True
    if detect_blur(img_bgr):
        return "Unusable_Blurry", "Motion blur detected", True
    if detect_ir_night(img_bgr):
        return "IR_Night", "Grayscale IR night image — converting to 3-channel", False
    return "Clean", "Good quality", False

def convert_ir_to_rgb(img_bgr):
    \"\"\"Convert IR grayscale to 3-channel so ResNet50 can process it.\"\"\"
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    rgb_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return rgb_3ch

print("Phase 1 triage functions ready.")
"""))

# ══════════════════════════════════════════════════════════════════════════════
# CELL 7 — Preprocessing
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
## Phase 2a — Preprocessing Chain
Handles all Sundarbans field conditions before detection runs.

| Step | Algorithm | Fixes |
|------|-----------|-------|
| 1 | CLAHE | Night / low-light images |
| 2 | Dark Channel Prior Dehazing | Fog / haze / rain scatter |
"""))
cells.append(code("""\
clahe_processor = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def apply_clahe(img_bgr):
    \"\"\"
    CLAHE — Contrast Limited Adaptive Histogram Equalisation.
    Works in LAB colour space (only L channel enhanced, colour preserved).
    \"\"\"
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_enhanced = clahe_processor.apply(l)
    enhanced   = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

def dehaze(img_bgr, patch_size=15, omega=0.95, t0=0.1):
    \"\"\"
    Dark Channel Prior Dehazing.
    Steps:
      1. Dark channel = min pixel across colour channels per patch
      2. Atmospheric light A = estimated from brightest 0.1% dark-channel pixels
      3. Transmission map t = 1 - omega * dark_channel_normalised
      4. Scene recovery: J = (I - A) / max(t, t0) + A
    \"\"\"
    img_float = img_bgr.astype(np.float64) / 255.0
    dark_ch   = np.min(img_float, axis=2)
    kernel    = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_ch   = cv2.erode(dark_ch.astype(np.float32), kernel).astype(np.float64)
    n_bright  = max(1, int(dark_ch.size * 0.001))
    flat_idx  = np.argsort(dark_ch.flatten())[-n_bright:]
    A         = np.max(img_float.reshape(-1, 3)[flat_idx], axis=0)
    A         = np.clip(A, 1e-6, 1.0)
    norm      = img_float / A
    dark_norm = np.min(norm, axis=2).astype(np.float32)
    dark_norm = cv2.erode(dark_norm, kernel).astype(np.float64)
    t         = np.clip(1.0 - omega * dark_norm, t0, 1.0)
    t3        = t[:, :, np.newaxis]
    J         = np.clip((img_float - A) / t3 + A, 0.0, 1.0)
    return (J * 255).astype(np.uint8)

def preprocess(img_bgr, is_ir=False):
    \"\"\"Full preprocessing: IR conversion → CLAHE → Dehazing.\"\"\"
    if is_ir:
        img_bgr = convert_ir_to_rgb(img_bgr)
    img = apply_clahe(img_bgr)
    img = dehaze(img)
    return img

print("Preprocessing functions ready.")
"""))

# ══════════════════════════════════════════════════════════════════════════════
# CELL 8 — Phase 2: Detection & Identity
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
## Phase 2b — Tiger Detection & Individual Identity
**YOLOv8** finds tigers in the image and draws bounding boxes.
**ResNet50** extracts a 2048-dimensional stripe fingerprint from each crop.
**Cosine Similarity** matches the fingerprint against the database.
"""))
cells.append(code("""\
def extract_fingerprint(crop_bgr):
    \"\"\"
    ResNet50 → 2048-dim stripe fingerprint.
    The [:-1] removes the 1000-class head.
    What remains: 2048 numbers describing the tiger's visual pattern.
    \"\"\"
    rgb     = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    tensor  = data_transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        vec = feature_extractor(tensor).squeeze().cpu().numpy()
    return vec

def match_or_enroll(fingerprint, update_db=True):
    \"\"\"
    Compare fingerprint against database using cosine similarity.
    Returns (tiger_id, match_score, is_new).

    update_db=True  → full body: update running average embedding
    update_db=False → partial body: read-only match, DB stays clean
    \"\"\"
    if not tiger_database:
        if update_db:
            tiger_id = "Tiger_001"
            tiger_database[tiger_id]        = fingerprint
            tiger_sighting_counts[tiger_id] = 1
            return tiger_id, 1.0, True
        else:
            return "Unknown_Partial", 0.0, True

    best_score, best_id = -1.0, None
    for enrolled_id, enrolled_vec in tiger_database.items():
        score = cosine_similarity(fingerprint.reshape(1,-1),
                                  enrolled_vec.reshape(1,-1))[0][0]
        if score > best_score:
            best_score, best_id = score, enrolled_id

    if best_score >= SIMILARITY_THRESHOLD:
        if update_db:
            n = tiger_sighting_counts[best_id]
            tiger_database[best_id] = (tiger_database[best_id]*n + fingerprint)/(n+1)
            tiger_sighting_counts[best_id] += 1
        return best_id, float(best_score), False
    else:
        if update_db:
            new_id = f"Tiger_{len(tiger_database)+1:03d}"
            tiger_database[new_id]        = fingerprint
            tiger_sighting_counts[new_id] = 1
            return new_id, float(best_score), True
        else:
            return best_id or "Unknown_Partial", float(best_score), True

print("Detection & identity functions ready.")
"""))

# ══════════════════════════════════════════════════════════════════════════════
# CELL 9 — Phase 3: Tiger Type Classification
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
## Phase 3 — Tiger Type Classification
**Color Morph** is detected using HSV colour analysis (no training needed).
**Subspecies** uses rule-based stripe density (ML classifier is next phase).

| Color Morph | Detection Method |
|-------------|-----------------|
| Orange Standard | Hue in orange range (10–25), normal saturation |
| White | High brightness (V>180), very low saturation (S<50) |
| Golden | Pale yellow hue (25–40), low saturation |
| Black (Pseudo-melanistic) | Very low brightness (V<60) |
| Snow White | Max brightness + near-zero saturation + near-zero variance |
"""))
cells.append(code("""\
def classify_color_morph(crop_bgr):
    \"\"\"Classify tiger color morph using HSV analysis.\"\"\"
    hsv  = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mean_h = float(np.mean(h))
    mean_s = float(np.mean(s))
    mean_v = float(np.mean(v))
    std_v  = float(np.std(v))

    if mean_v > 220 and mean_s < 20 and std_v < 15:
        return "Snow_White"
    if mean_v > 180 and mean_s < 50:
        return "White"
    if mean_v < 60:
        return "Black_Pseudomelanistic"
    if 25 <= mean_h <= 40 and mean_s < 120:
        return "Golden"
    return "Orange_Standard"

def classify_subspecies(crop_bgr):
    \"\"\"
    Subspecies via stripe density (Laplacian variance).
    Higher variance = more edge detail = denser stripes.
    Sumatran: densest stripes | Siberian: sparsest stripes
    Note: Replace with trained classifier once labelled data is available.
    \"\"\"
    gray    = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if lap_var > 800:
        return "Sumatran"
    elif lap_var < 300:
        return "Siberian"
    else:
        return "Bengal"

print("Phase 3 classification functions ready.")
"""))

# ══════════════════════════════════════════════════════════════════════════════
# CELL 10 — Run Pipeline
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("## Run Pipeline — Process All Images"))
cells.append(code("""\
results_log = []
all_images  = [
    p for p in Path(IMAGE_DIR).rglob("*")
    if p.suffix.lower() in SUPPORTED_EXTS
]

print(f"Found {len(all_images)} images in {IMAGE_DIR}")
print(f"Survey: {RANGE_NAME} | {BEAT_NAME}")
print("Starting TRACE pipeline...\\n")

skipped = 0

for img_path in all_images:

    # Load image
    img = cv2.imread(str(img_path))
    if img is None:
        try:
            pil = Image.open(img_path).convert("RGB")
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        except Exception:
            skipped += 1
            continue

    # ── PHASE 1: Triage ───────────────────────────────────────────────────────
    quality_bucket, quality_note, should_skip = triage_image(img)
    if should_skip:
        results_log.append({
            "Image_File": img_path.name, "Range": RANGE_NAME, "Beat": BEAT_NAME,
            "Image_Quality": quality_bucket, "Tigers_In_Frame": 0,
            "Identified_ID": "UNUSABLE", "Color_Morph": None, "Subspecies": None,
            "Status": f"Rejected — {quality_note}",
            "Match_Score": None, "Detection_Confidence": None,
        })
        skipped += 1
        continue

    # ── PHASE 2a: Preprocessing ───────────────────────────────────────────────
    is_ir = (quality_bucket == "IR_Night")
    img   = preprocess(img, is_ir=is_ir)

    # ── PHASE 2b: YOLO Detection ──────────────────────────────────────────────
    det_result = detector(str(img_path), verbose=False)[0]
    boxes      = det_result.boxes

    if len(boxes) == 0:
        results_log.append({
            "Image_File": img_path.name, "Range": RANGE_NAME, "Beat": BEAT_NAME,
            "Image_Quality": quality_bucket, "Tigers_In_Frame": 0,
            "Identified_ID": "NO_DETECTION", "Color_Morph": None, "Subspecies": None,
            "Status": "No animal detected (False Positive / empty frame)",
            "Match_Score": None, "Detection_Confidence": None,
        })
        skipped += 1
        continue

    img_area = img.shape[0] * img.shape[1]

    for box in boxes:
        det_conf  = float(box.conf[0])
        class_id  = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Filter 1: Animal class
        if class_id not in TIGER_PROXY_CLASSES:
            results_log.append({
                "Image_File": img_path.name, "Range": RANGE_NAME, "Beat": BEAT_NAME,
                "Image_Quality": quality_bucket, "Tigers_In_Frame": len(boxes),
                "Identified_ID": "FILTERED", "Color_Morph": None, "Subspecies": None,
                "Status": f"Filtered — wrong class (COCO {class_id})",
                "Match_Score": None, "Detection_Confidence": round(det_conf, 4),
            })
            continue

        # Filter 2: Confidence
        if det_conf < CONFIDENCE_THRESHOLD:
            results_log.append({
                "Image_File": img_path.name, "Range": RANGE_NAME, "Beat": BEAT_NAME,
                "Image_Quality": quality_bucket, "Tigers_In_Frame": len(boxes),
                "Identified_ID": "FILTERED", "Color_Morph": None, "Subspecies": None,
                "Status": f"Filtered — low confidence ({det_conf:.2f})",
                "Match_Score": None, "Detection_Confidence": round(det_conf, 4),
            })
            continue

        # Filter 3: Box size
        box_ratio = (x2-x1)*(y2-y1) / img_area
        if box_ratio < MIN_BOX_PARTIAL:
            results_log.append({
                "Image_File": img_path.name, "Range": RANGE_NAME, "Beat": BEAT_NAME,
                "Image_Quality": quality_bucket, "Tigers_In_Frame": len(boxes),
                "Identified_ID": "FILTERED", "Color_Morph": None, "Subspecies": None,
                "Status": f"Filtered — tail/fragment ({100*box_ratio:.1f}%)",
                "Match_Score": None, "Detection_Confidence": round(det_conf, 4),
            })
            continue

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # ── PHASE 2c: Identity ────────────────────────────────────────────────
        fingerprint = extract_fingerprint(crop)
        is_partial  = box_ratio < MIN_BOX_FULL

        if is_partial:
            tiger_id, match_score, is_new = match_or_enroll(fingerprint, update_db=False)
            status = ("Possible New — Partial Body" if is_new
                      else f"Possible Match — Partial Body (score {match_score:.2f})")
        else:
            tiger_id, match_score, is_new = match_or_enroll(fingerprint, update_db=True)
            status = "New Enrollment" if is_new else "Recaptured Match"

        # ── PHASE 3: Tiger Type ───────────────────────────────────────────────
        color_morph = classify_color_morph(crop)
        subspecies  = classify_subspecies(crop)

        results_log.append({
            "Image_File": img_path.name, "Range": RANGE_NAME, "Beat": BEAT_NAME,
            "Image_Quality": quality_bucket, "Tigers_In_Frame": len(boxes),
            "Identified_ID": tiger_id, "Color_Morph": color_morph,
            "Subspecies": subspecies, "Status": status,
            "Match_Score": round(match_score, 4),
            "Detection_Confidence": round(det_conf, 4),
        })

# Save
df = pd.DataFrame(results_log)
df.to_csv(OUTPUT_CSV, index=False)
save_database()
print("Pipeline complete. Results saved.")
"""))

# ══════════════════════════════════════════════════════════════════════════════
# CELL 11 — Final Report
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("## Results — Final Report"))
cells.append(code("""\
unusable  = len([r for r in results_log if r["Identified_ID"] == "UNUSABLE"])
filtered  = len([r for r in results_log if r["Identified_ID"] == "FILTERED"])
no_det    = len([r for r in results_log if r["Identified_ID"] == "NO_DETECTION"])
partial   = len([r for r in results_log if "Partial" in r.get("Status", "")])
processed = len([r for r in results_log if r["Identified_ID"] not in
                 ("NO_DETECTION","FILTERED","UNUSABLE")
                 and "Partial" not in r.get("Status","")])

print("=" * 60)
print("        TRACE — TIGER ENUMERATION REPORT")
print("=" * 60)
print(f"  Survey   : {RANGE_NAME} | {BEAT_NAME}")
print(f"  Total images         : {len(all_images)}")
print(f"\\n  PHASE 1 — IMAGE TRIAGE")
print(f"    Unusable images    : {unusable}")
print(f"    False positives    : {no_det}")
print(f"\\n  PHASE 2 — DETECTION & IDENTITY")
print(f"    Filtered           : {filtered}")
print(f"    Partial sightings  : {partial}")
print(f"    Valid detections   : {processed}")
print(f"    UNIQUE TIGERS      : {len(tiger_database)}")
print(f"\\n  PHASE 3 — TIGER TYPES")
morph_counts = {}
for r in results_log:
    m = r.get("Color_Morph")
    if m: morph_counts[m] = morph_counts.get(m,0)+1
for m,c in sorted(morph_counts.items(), key=lambda x:-x[1]):
    print(f"    {m:<28}: {c}")
print("=" * 60)

# Show results table
print("\\nDetailed results:")
display(df[df["Identified_ID"].str.startswith("Tiger", na=False)].head(20))
"""))

# ══════════════════════════════════════════════════════════════════════════════
# CELL 12 — Download Report
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("## Download Results CSV"))
cells.append(code("""\
from google.colab import files
files.download(OUTPUT_CSV)
print(f"Downloaded: {OUTPUT_CSV}")
"""))

# ══════════════════════════════════════════════════════════════════════════════
# BUILD NOTEBOOK
# ══════════════════════════════════════════════════════════════════════════════
nb.cells = cells
output_path = "TRACE_Tiger_Pipeline.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook saved: {output_path}")
