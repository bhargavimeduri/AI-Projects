"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          TRACE — Tiger Recognition And Census Engine  v5                   ║
║          EPAIB Batch 05 | Group 4 | IIM Lucknow                            ║
║          Real-world partner: Wildlife Dept, Govt of West Bengal            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  3-MODEL ARCHITECTURE (industry best practice)                             ║
║                                                                            ║
║  PHASE 1 — Image Triage (Model 1)                                          ║
║    Classify every image BEFORE any ML runs:                                ║
║    • IR/night grayscale detection                                           ║
║    • Overexposure / whiteout detection                                      ║
║    • Motion blur / underexposure detection                                  ║
║    → Buckets: Clean | Partial | Unusable | False Positive                  ║
║                                                                            ║
║  PHASE 2 — Tiger Detection & Individual ID (Model 2)                       ║
║    Step 1: YOLOv8  → Detect animals, filter non-tigers, count per frame    ║
║    Step 2: ResNet50 → Extract 2048-dim stripe fingerprint                  ║
║    Step 3: Cosine Similarity → Match identity or enroll new tiger          ║
║                                                                            ║
║  PHASE 3 — Tiger Type Classification (Model 3)                             ║
║    Color Morph: Orange / White / Golden / Black / Snow White               ║
║    Subspecies:  Bengal / Sumatran / Siberian / Malayan / S.China           ║
║    (rule-based HSV analysis — ML classifier in next phase)                 ║
║                                                                            ║
║  PREPROCESSING (applied before Phase 2)                                    ║
║    • CLAHE           → night / low-light images                            ║
║    • Dark Channel Prior dehazing → foggy / rainy / hazy images             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

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

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
TARGET_DIR  = BASE_DIR / "data" / "sample" / "tigers" / "train"
REPORTS_DIR    = BASE_DIR / "reports"
ANNOTATED_DIR  = REPORTS_DIR / "annotated_images"   # bounding box visualisations
REPORTS_DIR.mkdir(exist_ok=True)
ANNOTATED_DIR.mkdir(exist_ok=True)
OUTPUT_CSV  = REPORTS_DIR / "tiger_enumeration_results.csv"
DB_FILE     = REPORTS_DIR / "tiger_registry_db.json"

# ── Field Metadata (edit before each survey session) ──────────────────────
RANGE_NAME = "Sundarbans Tiger Reserve"
BEAT_NAME  = "Block-A / Sector-3"

# ── Phase 1: Image Triage thresholds ──────────────────────────────────────
BLUR_THRESHOLD        = 80.0   # Laplacian variance below this = blurry image
OVEREXPOSE_THRESHOLD  = 240    # Mean brightness above this = whiteout
UNDEREXPOSE_THRESHOLD = 20     # Mean brightness below this = blackout
IR_CHANNEL_DIFF       = 5      # Max std-dev between RGB channels = grayscale/IR

# ── Phase 2: Detection & Identity thresholds ──────────────────────────────
SIMILARITY_THRESHOLD    = 0.83  # Confirmed match — update DB embedding
PARTIAL_MATCH_THRESHOLD = 0.70  # Possible match (half-body) — log only
CONFIDENCE_THRESHOLD    = 0.35  # YOLO detection confidence minimum
MIN_BOX_FULL            = 0.05  # >= 5% of frame → full body → enroll/match
MIN_BOX_PARTIAL         = 0.03  # 3–5% of frame → half-body → log only
CROP_PAD                = 15    # pixels to pad around YOLO bounding box before crop
SUPPORTED_EXTS          = {".jpg", ".jpeg", ".png", ".webp"}

# ── Bounding box colours per colour morph (BGR) ───────────────────────────
MORPH_COLORS = {
    "Orange_Standard":        (0,   140, 255),   # orange
    "White":                  (200, 200, 255),   # pale blue-white
    "Black_Pseudomelanistic": (80,  80,  80),    # dark grey
    "Golden":                 (0,   215, 255),   # gold
    "Snow_White":             (255, 255, 255),   # white
}

# ── YOLO class filter ──────────────────────────────────────────────────────
# COCO has no "tiger" class. Proxies until YOLO is fine-tuned on tiger data:
#   15=cat  16=dog  17=horse  24=zebra (striped large animal)
# After fine-tuning: replace with {0} (or your assigned tiger class ID)
TIGER_PROXY_CLASSES = {15, 16, 17, 24}

# ── Device ─────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

print(f"[INFO] Device: {DEVICE.upper()}")

print("[INFO] Loading YOLOv8 detector (Phase 2 — Detection)...")
detector = YOLO("yolov8n.pt")

print("[INFO] Loading ResNet50 feature extractor (Phase 2 — Identity)...")
resnet_base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
feature_extractor = torch.nn.Sequential(
    *(list(resnet_base.children())[:-1])   # remove 1000-class head, keep 2048-dim features
).to(DEVICE)
feature_extractor.eval()

# Standard ImageNet preprocessing: Resize → CenterCrop (preserves aspect ratio)
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ══════════════════════════════════════════════════════════════════════════════
# TIGER DATABASE  (persistent across runs)
# ══════════════════════════════════════════════════════════════════════════════

tiger_database      = {}   # {"Tiger_001": np.array(2048,)}
tiger_sighting_counts = {} # {"Tiger_001": int}
results_log         = []

def load_database():
    """Load existing tiger registry from disk if available."""
    if DB_FILE.exists():
        try:
            with open(DB_FILE, "r") as f:
                raw = json.load(f)
            for tid, entry in raw.items():
                tiger_database[tid]       = np.array(entry["embedding"])
                tiger_sighting_counts[tid] = entry["sighting_count"]
            print(f"[INFO] Loaded {len(tiger_database)} existing tiger profiles.")
        except Exception as e:
            print(f"[WARN] Could not load database: {e} — starting fresh.")

def save_database():
    """Persist tiger registry to disk."""
    serialisable = {
        tid: {
            "embedding":      tiger_database[tid].tolist(),
            "sighting_count": tiger_sighting_counts[tid]
        }
        for tid in tiger_database
    }
    with open(DB_FILE, "w") as f:
        json.dump(serialisable, f, indent=2)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — IMAGE TRIAGE
# Classify every image into a quality bucket BEFORE any ML processing runs.
# Buckets: Clean | Partial | Unusable | False_Positive
# ══════════════════════════════════════════════════════════════════════════════

def detect_ir_night(img_bgr):
    """
    Detect IR / night grayscale images.
    Camera traps use infrared flash at night → black & white output.
    When all 3 RGB channels are nearly identical, image is grayscale.
    Returns True if IR/grayscale detected.
    """
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    diff_rg = np.std(rgb[:, :, 0] - rgb[:, :, 1])
    diff_rb = np.std(rgb[:, :, 0] - rgb[:, :, 2])
    return (diff_rg < IR_CHANNEL_DIFF) and (diff_rb < IR_CHANNEL_DIFF)

def detect_overexposure(img_bgr):
    """
    Detect overexposed / whiteout images.
    Happens when animal is too close to IR flash at night.
    Returns True if mean brightness > threshold.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray)) > OVEREXPOSE_THRESHOLD

def detect_underexposure(img_bgr):
    """
    Detect underexposed / blackout images.
    Happens when animal is too far, or dense canopy blocks all light.
    Returns True if mean brightness < threshold.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray)) < UNDEREXPOSE_THRESHOLD

def detect_blur(img_bgr):
    """
    Detect motion blur using Laplacian variance.
    Fast-moving animal + slow shutter = streaky, unrecognisable image.
    Low variance = sharp edges are absent = blurry.
    Returns True if image is too blurry to use.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var()) < BLUR_THRESHOLD

def triage_image(img_bgr):
    """
    Phase 1 — Full image quality assessment.
    Returns (quality_bucket, issue_description, should_skip).

    Quality Buckets:
        Clean        → pass to Phase 2 (detection)
        IR_Night     → grayscale IR image — still usable, noted
        Overexposed  → whiteout — unusable, skip
        Underexposed → blackout — unusable, skip
        Blurry       → motion blur — unusable, skip
    """
    if detect_overexposure(img_bgr):
        return "Unusable_Overexposed", "IR flash whiteout — mean brightness > 240", True

    if detect_underexposure(img_bgr):
        return "Unusable_Underexposed", "Blackout image — mean brightness < 20", True

    if detect_blur(img_bgr):
        return "Unusable_Blurry", f"Motion blur — Laplacian variance < {BLUR_THRESHOLD}", True

    if detect_ir_night(img_bgr):
        return "IR_Night", "Grayscale IR night image — processing with channel duplication", False

    return "Clean", "Good quality image", False

def convert_ir_to_rgb(img_bgr):
    """
    Convert IR / grayscale night image to 3-channel RGB.
    ResNet50 expects 3 channels — we duplicate the single channel.
    This preserves stripe contrast for embedding extraction.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    rgb_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return rgb_3ch

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — PREPROCESSING CHAIN
# Applied to all usable images before YOLO detection.
# ══════════════════════════════════════════════════════════════════════════════

clahe_processor = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def apply_clahe(img_bgr):
    """
    CLAHE — Contrast Limited Adaptive Histogram Equalisation.
    Fixes: night images, low-light camera trap shots.
    Operates in LAB colour space (only L channel enhanced — preserves colour).
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_enhanced = clahe_processor.apply(l)
    enhanced   = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

def dehaze(img_bgr, patch_size=15, omega=0.95, t0=0.1):
    """
    Dark Channel Prior Dehazing — removes fog, haze, and rain scatter.

    How it works:
      1. Dark Channel: in non-sky patches, at least one colour channel has
         very low intensity. Fog raises this — high dark channel = hazy.
      2. Atmospheric light A: estimated from brightest 0.1% dark-channel pixels.
      3. Transmission map t: how much original scene light reaches the camera.
      4. Scene recovery: J = (I - A) / max(t, t0) + A

    omega=0.95 → remove 95% of haze (keep a slight depth cue)
    t0=0.10   → minimum transmission floor (prevents divide-by-near-zero)
    """
    img_float = img_bgr.astype(np.float64) / 255.0

    # Step 1 — Dark channel
    dark_ch = np.min(img_float, axis=2)
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_ch = cv2.erode(dark_ch.astype(np.float32), kernel).astype(np.float64)

    # Step 2 — Atmospheric light
    n_bright = max(1, int(dark_ch.size * 0.001))
    flat_idx = np.argsort(dark_ch.flatten())[-n_bright:]
    A        = np.max(img_float.reshape(-1, 3)[flat_idx], axis=0)
    A        = np.clip(A, 1e-6, 1.0)

    # Step 3 — Transmission map
    norm      = img_float / A
    dark_norm = np.min(norm, axis=2).astype(np.float32)
    dark_norm = cv2.erode(dark_norm, kernel).astype(np.float64)
    t         = np.clip(1.0 - omega * dark_norm, t0, 1.0)

    # Step 4 — Scene recovery
    t3 = t[:, :, np.newaxis]
    J  = (img_float - A) / t3 + A
    J  = np.clip(J, 0.0, 1.0)
    return (J * 255).astype(np.uint8)

def preprocess(img_bgr, is_ir=False):
    """
    Full preprocessing chain for Sundarbans camera trap images.
    1. IR conversion (if night grayscale image)
    2. CLAHE  — fixes night / low-light contrast
    3. Dehaze — fixes foggy / hazy / rainy scatter
    """
    if is_ir:
        img_bgr = convert_ir_to_rgb(img_bgr)
    img = apply_clahe(img_bgr)
    img = dehaze(img)
    return img

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — TIGER DETECTION (YOLOv8) & INDIVIDUAL IDENTITY (ResNet50)
# ══════════════════════════════════════════════════════════════════════════════

def extract_fingerprint(crop_bgr):
    """
    ResNet50 → 2048-dimensional stripe fingerprint.
    The last classification layer ([:-1]) was removed — we keep the feature
    vector, not the 1000-class prediction. Stripes are unique per individual.
    """
    rgb     = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    tensor  = data_transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        vec = feature_extractor(tensor).squeeze().cpu().numpy()
    return vec

def match_or_enroll(fingerprint, update_db=True):
    """
    Compare fingerprint against tiger database using cosine similarity.
    Returns (tiger_id, match_score, is_new).

    update_db=True  → full body: update running average, increment sighting count
    update_db=False → partial body: match only, never modify the database
                      (protects DB quality from noisy partial embeddings)
    """
    if not tiger_database:
        if update_db:
            tiger_id = "Tiger_001"
            tiger_database[tiger_id]       = fingerprint
            tiger_sighting_counts[tiger_id] = 1
            return tiger_id, 1.0, True
        else:
            return "Unknown_Partial", 0.0, True

    best_score = -1.0
    best_id    = None
    for enrolled_id, enrolled_vec in tiger_database.items():
        score = cosine_similarity(
            fingerprint.reshape(1, -1),
            enrolled_vec.reshape(1, -1)
        )[0][0]
        if score > best_score:
            best_score = score
            best_id    = enrolled_id

    if best_score >= SIMILARITY_THRESHOLD:
        if update_db:
            # Running average embedding — improves with each sighting
            n = tiger_sighting_counts[best_id]
            tiger_database[best_id] = (tiger_database[best_id] * n + fingerprint) / (n + 1)
            tiger_sighting_counts[best_id] += 1
        return best_id, float(best_score), False
    else:
        if update_db:
            new_id = f"Tiger_{len(tiger_database) + 1:03d}"
            tiger_database[new_id]       = fingerprint
            tiger_sighting_counts[new_id] = 1
            return new_id, float(best_score), True
        else:
            return best_id if best_id else "Unknown_Partial", float(best_score), True

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — TIGER TYPE CLASSIFICATION
# Rule-based colour morph detection using HSV analysis.
# ML-based subspecies classifier is Phase 2 next step (needs labelled data).
# ══════════════════════════════════════════════════════════════════════════════

def classify_color_morph(crop_bgr):
    """
    Classify tiger color morph from cropped tiger image using HSV analysis.

    Color Morphs:
      Orange      → dominant hue 10–25 (orange range), normal saturation
      White       → high brightness (V > 200), very low saturation (S < 50)
      Golden      → pale yellow hue (25–40), low-medium saturation
      Black       → very low brightness (V < 60), high stripe density
      Snow_White  → maximum brightness, near-zero saturation AND near-zero variance

    Returns: color morph string
    """
    hsv  = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    mean_h = float(np.mean(h))
    mean_s = float(np.mean(s))
    mean_v = float(np.mean(v))
    std_v  = float(np.std(v))

    # Snow White — pure white, stripes invisible
    if mean_v > 220 and mean_s < 20 and std_v < 15:
        return "Snow_White"

    # White Tiger — white coat, chocolate stripes
    if mean_v > 180 and mean_s < 50:
        return "White"

    # Black Tiger (Pseudo-melanistic) — very dark, stripes dominate
    if mean_v < 60:
        return "Black_Pseudomelanistic"

    # Golden Tiger — pale gold/blonde
    if 25 <= mean_h <= 40 and mean_s < 120:
        return "Golden"

    # Standard Orange Tiger — default
    return "Orange_Standard"

def classify_subspecies_placeholder(crop_bgr):
    """
    Subspecies classification placeholder.
    Full ML classifier requires labelled subspecies training data.

    Rule-based approximation by stripe density:
      High density (Laplacian variance > 800)  → Sumatran
      Low density  (Laplacian variance < 300)  → Siberian
      Medium                                   → Bengal (most common in India)

    Replace with trained ResNet50 classifier head in Phase 2 build.
    """
    gray     = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    lap_var  = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if lap_var > 800:
        return "Sumatran (high stripe density)"
    elif lap_var < 300:
        return "Siberian (low stripe density)"
    else:
        return "Bengal (medium stripe density — most likely in India)"

# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE — Load DB + Process All Images
# ══════════════════════════════════════════════════════════════════════════════

load_database()

all_images = [
    p for p in TARGET_DIR.rglob("*")
    if p.suffix.lower() in SUPPORTED_EXTS
]

print(f"\n[INFO] Found {len(all_images)} images in {TARGET_DIR}")
print(f"[INFO] Survey location: {RANGE_NAME} | {BEAT_NAME}")
print("[INFO] Starting 3-phase pipeline...\n")

skipped = 0

for img_path in all_images:

    # ── Load image ────────────────────────────────────────────────────────────
    img = cv2.imread(str(img_path))
    if img is None:
        try:
            pil = Image.open(img_path).convert("RGB")
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        except Exception:
            skipped += 1
            continue

    # ── PHASE 1: Image Triage ─────────────────────────────────────────────────
    quality_bucket, quality_note, should_skip = triage_image(img)

    if should_skip:
        results_log.append({
            "Image_File":           img_path.name,
            "Range":                RANGE_NAME,
            "Beat":                 BEAT_NAME,
            "Image_Quality":        quality_bucket,
            "Tigers_In_Frame":      0,
            "Identified_ID":        "UNUSABLE",
            "Color_Morph":          None,
            "Subspecies":           None,
            "Status":               f"Rejected — {quality_note}",
            "Match_Score":          None,
            "Detection_Confidence": None,
        })
        skipped += 1
        continue

    # ── PHASE 2a: Preprocessing ───────────────────────────────────────────────
    is_ir = (quality_bucket == "IR_Night")
    img   = preprocess(img, is_ir=is_ir)

    # ── PHASE 2b: YOLO Detection ──────────────────────────────────────────────
    # Pass the preprocessed numpy array — YOLO sees the enhanced image,
    # not the original file (fixes detection quality on foggy/dark images)
    detection_results = detector(img, verbose=False)[0]
    boxes = detection_results.boxes

    # Prepare annotated image canvas for bounding box visualisation
    annotated_img  = img.copy()
    valid_det_count = 0   # count of boxes that pass all filters

    if len(boxes) == 0:
        results_log.append({
            "Image_File":           img_path.name,
            "Range":                RANGE_NAME,
            "Beat":                 BEAT_NAME,
            "Image_Quality":        quality_bucket,
            "Tigers_In_Frame":      0,
            "Identified_ID":        "NO_DETECTION",
            "Color_Morph":          None,
            "Subspecies":           None,
            "Status":               "Skipped — no animal detected (False Positive / empty frame)",
            "Match_Score":          None,
            "Detection_Confidence": None,
        })
        skipped += 1
        continue

    img_area = img.shape[0] * img.shape[1]

    for box in boxes:
        det_confidence = float(box.conf[0])
        class_id       = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # ── Filter 1: Animal class ────────────────────────────────────────────
        if class_id not in TIGER_PROXY_CLASSES:
            results_log.append({
                "Image_File":           img_path.name,
                "Range":                RANGE_NAME,
                "Beat":                 BEAT_NAME,
                "Image_Quality":        quality_bucket,
                "Tigers_In_Frame":      len(boxes),
                "Identified_ID":        "FILTERED",
                "Color_Morph":          None,
                "Subspecies":           None,
                "Status":               f"Filtered — wrong class (COCO class {class_id}, not a tiger proxy)",
                "Match_Score":          None,
                "Detection_Confidence": round(det_confidence, 4),
            })
            continue

        # ── Filter 2: Confidence ──────────────────────────────────────────────
        if det_confidence < CONFIDENCE_THRESHOLD:
            results_log.append({
                "Image_File":           img_path.name,
                "Range":                RANGE_NAME,
                "Beat":                 BEAT_NAME,
                "Image_Quality":        quality_bucket,
                "Tigers_In_Frame":      len(boxes),
                "Identified_ID":        "FILTERED",
                "Color_Morph":          None,
                "Subspecies":           None,
                "Status":               f"Filtered — low confidence ({det_confidence:.2f} < {CONFIDENCE_THRESHOLD})",
                "Match_Score":          None,
                "Detection_Confidence": round(det_confidence, 4),
            })
            continue

        # ── Filter 3: Box size ────────────────────────────────────────────────
        box_area  = (x2 - x1) * (y2 - y1)
        box_ratio = box_area / img_area

        if box_ratio < MIN_BOX_PARTIAL:
            results_log.append({
                "Image_File":           img_path.name,
                "Range":                RANGE_NAME,
                "Beat":                 BEAT_NAME,
                "Image_Quality":        quality_bucket,
                "Tigers_In_Frame":      len(boxes),
                "Identified_ID":        "FILTERED",
                "Color_Morph":          None,
                "Subspecies":           None,
                "Status":               f"Filtered — tail/fragment ({100*box_ratio:.1f}% < 3% of frame)",
                "Match_Score":          None,
                "Detection_Confidence": round(det_confidence, 4),
            })
            continue

        # ── Visibility tier ───────────────────────────────────────────────────
        # Aligns with Colab script's full / partial / trace labelling
        if box_ratio >= MIN_BOX_FULL:
            visibility = "Full_Body"
        else:
            visibility = "Partial_Body"   # 3–5% already passed MIN_BOX_PARTIAL above

        # ── Padded crop — prevents edge artifacts on border detections ────────
        img_h, img_w = img.shape[:2]
        px1 = max(0, x1 - CROP_PAD)
        py1 = max(0, y1 - CROP_PAD)
        px2 = min(img_w, x2 + CROP_PAD)
        py2 = min(img_h, y2 + CROP_PAD)

        crop = img[py1:py2, px1:px2]
        if crop.size == 0:
            continue

        # ── PHASE 2c: Individual Identity ─────────────────────────────────────
        fingerprint = extract_fingerprint(crop)
        is_partial  = (visibility == "Partial_Body")

        if is_partial:
            tiger_id, match_score, is_new = match_or_enroll(fingerprint, update_db=False)
            if is_new:
                status = "Possible New — Partial Body (not enrolled)"
            elif match_score >= SIMILARITY_THRESHOLD:
                status = "Possible Match — Partial Body (DB not updated)"
            else:
                status = f"Uncertain — Partial Body (best score {match_score:.2f})"
        else:
            tiger_id, match_score, is_new = match_or_enroll(fingerprint, update_db=True)
            status = "New Enrollment" if is_new else "Recaptured Match"

        # ── PHASE 3: Tiger Type Classification ────────────────────────────────
        color_morph = classify_color_morph(crop)
        subspecies  = classify_subspecies_placeholder(crop)
        valid_det_count += 1

        # ── Bounding box visualisation (added v5) ─────────────────────────────
        box_color = MORPH_COLORS.get(color_morph, (0, 255, 0))
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), box_color, 3)

        # Label line 1: Tiger ID + morph
        label1 = f"{tiger_id} | {color_morph}"
        # Label line 2: confidence + visibility + match score
        label2 = (f"Det:{det_confidence:.0%}  "
                  f"Match:{match_score:.2f}  "
                  f"[{visibility}]")

        (lw, lh), _ = cv2.getTextSize(label1, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        bg_y1 = max(0, y1 - lh * 2 - 14)
        cv2.rectangle(annotated_img,
                      (x1, bg_y1), (x1 + max(lw + 8, 220), y1),
                      box_color, -1)

        text_color = (0, 0, 0) if color_morph in ("White", "Snow_White", "Golden") \
                     else (255, 255, 255)
        cv2.putText(annotated_img, label1,
                    (x1 + 4, y1 - lh - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 2)
        cv2.putText(annotated_img, label2,
                    (x1 + 4, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, text_color, 1)

        results_log.append({
            "Image_File":           img_path.name,
            "Range":                RANGE_NAME,
            "Beat":                 BEAT_NAME,
            "Image_Quality":        quality_bucket,
            "Tigers_In_Frame":      valid_det_count,   # valid detections only
            "Visibility":           visibility,
            "Identified_ID":        tiger_id,
            "Color_Morph":          color_morph,
            "Subspecies":           subspecies,
            "Status":               status,
            "Match_Score":          round(match_score, 4),
            "Detection_Confidence": round(det_confidence, 4),
        })

    # ── Save annotated image if any valid detections ───────────────────────
    if valid_det_count > 0:
        out_path = ANNOTATED_DIR / img_path.name
        cv2.imwrite(str(out_path), annotated_img)

# ══════════════════════════════════════════════════════════════════════════════
# SAVE OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

df = pd.DataFrame(results_log)
df.to_csv(OUTPUT_CSV, index=False)
save_database()

# ══════════════════════════════════════════════════════════════════════════════
# FINAL REPORT
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("              TRACE — TIGER ENUMERATION REPORT")
print("=" * 65)
print(f"  Survey location        : {RANGE_NAME} | {BEAT_NAME}")

unusable  = len([r for r in results_log if r["Identified_ID"] == "UNUSABLE"])
filtered  = len([r for r in results_log if r["Identified_ID"] == "FILTERED"])
no_det    = len([r for r in results_log if r["Identified_ID"] == "NO_DETECTION"])
partial   = len([r for r in results_log if "Partial" in r.get("Status", "")])
processed = len([r for r in results_log if r["Identified_ID"] not in
                 ("NO_DETECTION", "FILTERED", "UNUSABLE")
                 and "Partial" not in r.get("Status", "")])

print(f"\n  PHASE 1 — IMAGE TRIAGE")
print(f"    Total images           : {len(all_images)}")
print(f"    Unusable (blur/white/  ")
print(f"    black/overexposed)     : {unusable}")
print(f"    False positives        : {no_det}  (no animal detected)")
print(f"\n  PHASE 2 — DETECTION & IDENTITY")
print(f"    Detections filtered    : {filtered}  (wrong class / low conf / fragment)")
print(f"    Partial-body sightings : {partial}  (logged — DB protected)")
print(f"    Valid full detections  : {processed}")
print(f"    UNIQUE TIGERS FOUND    : {len(tiger_database)}")

if tiger_sighting_counts:
    print(f"\n  PHASE 3 — TIGER TYPE SUMMARY")
    morph_counts = {}
    for r in results_log:
        m = r.get("Color_Morph")
        if m:
            morph_counts[m] = morph_counts.get(m, 0) + 1
    for morph, count in sorted(morph_counts.items(), key=lambda x: -x[1]):
        print(f"    {morph:<30}: {count} detection(s)")

print(f"\n  OUTPUT FILES")
print(f"    Database               : {DB_FILE}")
print(f"    Report CSV             : {OUTPUT_CSV}")
print(f"    Annotated images       : {ANNOTATED_DIR}")
print("=" * 65)

print("\nPer-tiger sighting counts:")
for tid, count in sorted(tiger_sighting_counts.items()):
    print(f"  {tid}: {count} sighting(s)")

print("\nSample results (first 10 valid detections):")
valid = df[df["Identified_ID"].str.startswith("Tiger", na=False)]
if not valid.empty:
    print(valid.head(10).to_string(index=False))
