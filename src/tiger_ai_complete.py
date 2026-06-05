"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   TIGER AI SYSTEM — Complete Local Version (Fixed from Colab)              ║
║   EPAIB Batch 05 | Group 4 | IIM Lucknow                                   ║
║                                                                              ║
║   Original: copy_of_tiger_ai_system_complete.py (Google Colab)             ║
║   Fixed   : Removed Colab dependencies, added TRACE features               ║
║                                                                              ║
║   WHAT WAS FIXED vs ORIGINAL COLAB SCRIPT                                  ║
║   ─────────────────────────────────────────────────────────────────────     ║
║   ✅ Removed  google.colab imports (won't crash locally)                    ║
║   ✅ Removed  nvidia-smi SystemExit (works on CPU too)                      ║
║   ✅ Replaced trained YOLO model → pretrained YOLOv8n (runs immediately)   ║
║   ✅ Replaced EfficientNetB3 classifier → ResNet50 + cosine similarity      ║
║   ✅ Added    Individual tiger identity (Tiger_001, Tiger_002 …)            ║
║   ✅ Added    Identity DB persistence (JSON save/load)                      ║
║   ✅ Added    CLAHE preprocessing (contrast for low-light images)           ║
║   ✅ Added    Dark Channel Prior dehazing (fog/rain/haze)                   ║
║   ✅ Added    IR/Night image detection & handling                           ║
║   ✅ Added    Image triage with skip logic (blur/over/underexpose)          ║
║   ✅ Added    Partial body DB protection (running average embedding)        ║
║   ✅ Added    Grad-CAM explainability on detected crops                     ║
║   ✅ Added    Crop padding (15px) — prevents edge artefacts                 ║
║   ✅ Added    Visibility label: Full_Body / Partial_Body / Corner_Trace     ║
║   ✅ Added    Shadow detection — tiger shadow ≠ second tiger               ║
║   ✅ Added    Corner trace detection — partial tiger at frame edges        ║
║   ✅ Added    Inter-box overlap guard — no duplicate counting              ║
║   ✅ Kept     Bounding box visualisation with colour-coded morph boxes      ║
║   ✅ Kept     Population report with summary statistics                     ║
║   ✅ Kept     analyze_image() — clean single-image analysis function        ║
║                                                                              ║
║   USAGE                                                                      ║
║     # Full batch run                                                         ║
║     py -3 src/tiger_ai_complete.py                                          ║
║                                                                              ║
║     # Single image                                                           ║
║     py -3 src/tiger_ai_complete.py --image tiger_01.jpg                    ║
║                                                                              ║
║     # With Grad-CAM                                                          ║
║     py -3 src/tiger_ai_complete.py --gradcam                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import json
import argparse
import warnings
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — SETUP & CONFIGURATION
# (Replaces Colab Cells 1.1–1.3: removed google.colab, nvidia-smi, Drive mount)
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR       = Path(__file__).resolve().parent.parent
DATA_DIR       = BASE_DIR / "data" / "sample" / "tigers" / "train"
REPORTS_DIR    = BASE_DIR / "reports"
ANNOTATED_DIR  = REPORTS_DIR / "annotated_images"
GRADCAM_DIR    = REPORTS_DIR / "gradcam_outputs"
DB_FILE        = REPORTS_DIR / "tiger_registry_db.json"
OUTPUT_CSV     = REPORTS_DIR / "tiger_population_report.csv"
REPORT_TXT     = REPORTS_DIR / "tiger_population_report.txt"

for d in [REPORTS_DIR, ANNOTATED_DIR, GRADCAM_DIR]:
    d.mkdir(exist_ok=True)

# ── Survey metadata ────────────────────────────────────────────────────────
RANGE_NAME = "Sundarbans Tiger Reserve"
BEAT_NAME  = "Block-A / Sector-3"

# ── Device ─────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device : {DEVICE.upper()}")

# ── Supported formats ──────────────────────────────────────────────────────
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ── Tiger type display names & bounding box colours (BGR) ─────────────────
TIGER_TYPES = {
    "Orange_Standard":        {"display": "Orange Tiger",      "color": (0,   140, 255)},
    "White":                  {"display": "White Tiger",       "color": (200, 200, 255)},
    "Black_Pseudomelanistic": {"display": "Black Tiger",       "color": (60,  60,  60) },
    "Golden":                 {"display": "Golden Tiger",      "color": (0,   215, 255)},
    "Snow_White":             {"display": "Snow White Tiger",  "color": (255, 255, 255)},
}

# ── Detection thresholds ───────────────────────────────────────────────────
CONFIDENCE_THRESHOLD    = 0.35
IOU_THRESHOLD           = 0.50   # NMS — increase if seeing duplicate boxes
MIN_BOX_FULL            = 0.05   # ≥ 5% of frame → full body
MIN_BOX_PARTIAL         = 0.03   # 3–5% → partial body
CROP_PAD                = 15     # pixel padding around crop
SIMILARITY_THRESHOLD    = 0.83   # cosine similarity — same tiger
PARTIAL_MATCH_THRESHOLD = 0.70

# ── YOLO proxy classes (COCO — no tiger class exists) ─────────────────────
# 15=cat  16=dog  17=horse  24=zebra
# After fine-tuning on tiger data, replace with: TIGER_PROXY_CLASSES = {0}
TIGER_PROXY_CLASSES = {15, 16, 17, 24}

# ── Image triage thresholds ────────────────────────────────────────────────
BLUR_THRESHOLD        = 80.0
OVEREXPOSE_THRESHOLD  = 240
UNDEREXPOSE_THRESHOLD = 20
IR_CHANNEL_DIFF       = 5


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — IMAGE QUALITY TRIAGE
# (Replaces Colab Cell 2.1 — adds skip logic, not just a scan)
# ══════════════════════════════════════════════════════════════════════════════

def detect_ir_night(img_bgr):
    """IR/night grayscale: all 3 RGB channels nearly identical → grayscale image."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    return (np.std(rgb[:,:,0] - rgb[:,:,1]) < IR_CHANNEL_DIFF and
            np.std(rgb[:,:,0] - rgb[:,:,2]) < IR_CHANNEL_DIFF)

def triage_image(img_bgr):
    """
    Classify image quality before any ML runs (cheapest filter first).
    Returns (quality_bucket, reason, should_skip).

    Buckets : Clean | IR_Night | Unusable_Blurry |
              Unusable_Overexposed | Unusable_Underexposed
    """
    gray       = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if brightness > OVEREXPOSE_THRESHOLD:
        return "Unusable_Overexposed",  f"Mean brightness {brightness:.0f} > {OVEREXPOSE_THRESHOLD}", True
    if brightness < UNDEREXPOSE_THRESHOLD:
        return "Unusable_Underexposed", f"Mean brightness {brightness:.0f} < {UNDEREXPOSE_THRESHOLD}", True
    if blur_score < BLUR_THRESHOLD:
        return "Unusable_Blurry",       f"Laplacian variance {blur_score:.1f} < {BLUR_THRESHOLD}", True
    if detect_ir_night(img_bgr):
        return "IR_Night", "Grayscale IR — processing with channel duplication", False

    return "Clean", f"Good quality (brightness={brightness:.0f}, blur={blur_score:.0f})", False


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — PREPROCESSING CHAIN
# (Missing from Colab — only had brightness boost; added CLAHE + DCP)
# ══════════════════════════════════════════════════════════════════════════════

_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def apply_clahe(img_bgr):
    """
    CLAHE on LAB L-channel only (preserves colour, enhances contrast).
    Fixes: dark camera trap images, low-light night shots.
    """
    lab            = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b        = cv2.split(lab)
    enhanced       = cv2.merge([_clahe.apply(l), a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

def dehaze(img_bgr, patch_size=15, omega=0.95, t0=0.1):
    """
    Dark Channel Prior dehazing — removes fog, rain scatter, haze.
    Critical for Sundarbans monsoon-season camera trap images.
    """
    img_f  = img_bgr.astype(np.float64) / 255.0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))

    dark_ch = cv2.erode(np.min(img_f, axis=2).astype(np.float32), kernel).astype(np.float64)
    n_bright= max(1, int(dark_ch.size * 0.001))
    A       = np.clip(np.max(img_f.reshape(-1, 3)[np.argsort(dark_ch.flatten())[-n_bright:]], axis=0), 1e-6, 1.0)

    dark_norm = cv2.erode((np.min(img_f / A, axis=2)).astype(np.float32), kernel).astype(np.float64)
    t         = np.clip(1.0 - omega * dark_norm, t0, 1.0)[:, :, np.newaxis]

    return np.clip(((img_f - A) / t + A) * 255, 0, 255).astype(np.uint8)

def preprocess(img_bgr, is_ir=False):
    """Full preprocessing chain: IR conversion → CLAHE → Dehaze."""
    if is_ir:
        gray   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        img_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return dehaze(apply_clahe(img_bgr))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — MODEL LOADING
# (Replaces Colab Cells 3.2/4.2/4.3 — no training needed, pretrained models)
# ══════════════════════════════════════════════════════════════════════════════

print("[INFO] Loading YOLOv8n detector …")
detector = YOLO("yolov8n.pt")

print("[INFO] Loading ResNet50 feature extractor …")
_resnet_base      = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
feature_extractor = torch.nn.Sequential(*list(_resnet_base.children())[:-1]).to(DEVICE)
feature_extractor.eval()

# ImageNet standard transform (CenterCrop preserves aspect ratio)
_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Full ResNet50 (with head) — used only for Grad-CAM
_resnet_full = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(DEVICE)
_resnet_full.eval()

print("[INFO] Models loaded.")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — INDIVIDUAL TIGER IDENTITY DATABASE
# (Missing from Colab entirely — the core census capability)
# ══════════════════════════════════════════════════════════════════════════════

tiger_db     = {}   # {"Tiger_001": np.array(2048,)}
tiger_counts = {}   # {"Tiger_001": int}  — number of full-body sightings

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
    """ResNet50 → 2048-dim stripe fingerprint (head removed)."""
    pil    = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    tensor = _transform(pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return feature_extractor(tensor).squeeze().cpu().numpy()

def match_or_enroll(fingerprint, update_db=True):
    """
    Cosine similarity match against DB.
    update_db=True  → full body: update running average, increment count
    update_db=False → partial:   match only, never modify DB
    """
    if not tiger_db:
        if update_db:
            tid = "Tiger_001"
            tiger_db[tid]     = fingerprint
            tiger_counts[tid] = 1
            return tid, 1.0, True
        return "Unknown_Partial", 0.0, True

    scores   = {tid: cosine_similarity(fingerprint.reshape(1,-1),
                                        vec.reshape(1,-1))[0][0]
                for tid, vec in tiger_db.items()}
    best_id  = max(scores, key=scores.get)
    best_scr = float(scores[best_id])

    if best_scr >= SIMILARITY_THRESHOLD:
        if update_db:
            n = tiger_counts[best_id]
            tiger_db[best_id]     = (tiger_db[best_id] * n + fingerprint) / (n + 1)
            tiger_counts[best_id] += 1
        return best_id, best_scr, False

    if update_db:
        new_id = f"Tiger_{len(tiger_db)+1:03d}"
        tiger_db[new_id]     = fingerprint
        tiger_counts[new_id] = 1
        return new_id, best_scr, True

    return best_id, best_scr, True


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — TIGER TYPE CLASSIFICATION
# (Replaces Colab EfficientNetB3 classifier — uses HSV, no training needed)
# ══════════════════════════════════════════════════════════════════════════════

def classify_color_morph(crop_bgr):
    """
    HSV-based colour morph classification — no training data required.
    Rules calibrated on known tiger colour ranges.
    """
    h, s, v = cv2.split(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV))
    mh, ms, mv, sv = np.mean(h), np.mean(s), np.mean(v), np.std(v)

    if mv > 220 and ms < 20  and sv < 15:  return "Snow_White"
    if mv > 180 and ms < 50:               return "White"
    if mv < 60:                             return "Black_Pseudomelanistic"
    if 25 <= mh <= 40 and ms < 120:        return "Golden"
    return "Orange_Standard"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — GRAD-CAM EXPLAINABILITY
# (Missing from Colab — added from TRACE gradcam_check.py)
# "A model you can't explain is a model you can't trust." — Prof. Mahesh Balan
# ══════════════════════════════════════════════════════════════════════════════

_fmaps, _grads = {}, {}

def _save_fmaps(m, i, o): _fmaps["l4"] = o
def _save_grads(m, gi, go): _grads["l4"] = go[0]

_resnet_full.layer4.register_forward_hook(_save_fmaps)
_resnet_full.layer4.register_full_backward_hook(_save_grads)

def generate_gradcam(crop_bgr, save_path, label=""):
    """
    Grad-CAM on ResNet50 layer4 — shows WHERE the model looks on the tiger crop.
    Good result: heatmap on stripes/body.  Bad result: heatmap on background.
    Saves side-by-side: Original | Heatmap | Overlay.
    """
    rgb    = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    tensor = _transform(Image.fromarray(rgb)).unsqueeze(0).to(DEVICE)
    tensor.requires_grad_(True)

    _resnet_full.zero_grad()
    out   = _resnet_full(tensor)
    out[0, out.argmax(dim=1).item()].backward()

    weights = _grads["l4"].mean(dim=[2, 3], keepdim=True)
    cam     = torch.relu((weights * _fmaps["l4"]).sum(dim=1).squeeze())
    cam     = cam.detach().cpu().numpy()
    if cam.max() > 0:
        cam /= cam.max()

    h, w        = crop_bgr.shape[:2]
    heatmap     = cv2.applyColorMap(
                    cv2.resize((cam * 255).astype(np.uint8), (w, h)),
                    cv2.COLORMAP_JET)
    overlay     = cv2.addWeighted(crop_bgr, 0.5, heatmap, 0.5, 0)

    if label:
        cv2.putText(overlay, label, (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

    header = np.zeros((28, w*3, 3), dtype=np.uint8)
    for i, txt in enumerate(["Original", "Grad-CAM", "Overlay"]):
        cv2.putText(header, txt, (i*w + w//4, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

    cv2.imwrite(str(save_path),
                np.vstack([header, np.hstack([crop_bgr, heatmap, overlay])]))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7.5 — SHADOW, CORNER TRACE & OVERLAP GUARDS
# Handles camera trap artefacts unique to wildlife monitoring:
#   1. Tiger shadow cast on ground (common in afternoon sidelight)
#   2. Partial tiger entering/leaving frame at image corners
#   3. Duplicate box from shadow spatially close to real tiger
# ══════════════════════════════════════════════════════════════════════════════

def is_shadow_crop(crop_bgr):
    """
    Returns True if crop is likely a tiger's SHADOW, not a real tiger.

    Logic
    ─────
    Real tigers have coloured stripes (orange/black) → mean HSV saturation > 40.
    Shadows are near-achromatic dark blobs: low saturation AND low brightness.

    Threshold rationale
    ───────────────────
    • sat < 30  — essentially grayscale; true orange fur has sat ≈ 120–180
    • val < 85  — dim; shadows absorb most incident light

    NOTE: IR/night images arrive as 3-channel grayscale (already converted by
          preprocess()), so their saturation is also ~0. However those images
          still have a large bright V channel (reflected IR). The val < 85 guard
          prevents false-positive shadow rejection on IR crops.
    """
    if crop_bgr.size == 0:
        return False
    hsv    = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    mean_s = float(np.mean(hsv[:, :, 1]))   # 0 = grey, 255 = vivid colour
    mean_v = float(np.mean(hsv[:, :, 2]))   # 0 = black, 255 = white
    return mean_s < 30 and mean_v < 85


def is_corner_trace(x1, y1, x2, y2, img_w, img_h, margin=12):
    """
    Returns True if the bounding box is at the image border.

    Camera traps at fixed positions often capture tigers entering or leaving
    frame — only a stripe patch at the corner is visible. These crops:
      • Are labelled 'Corner_Trace' visibility
      • Do NOT update the identity DB  (corrupted running average otherwise)
      • Still attempt a match (may confirm a known individual)
    """
    return (x1 <= margin or y1 <= margin or
            x2 >= img_w - margin or y2 >= img_h - margin)


def check_box_overlap(accepted_boxes, new_box, iou_threshold=0.45):
    """
    Returns True if new_box overlaps any already-accepted box above iou_threshold.

    Purpose: prevent a tiger shadow (cast nearby) from being counted as a
    second tiger. YOLO's own NMS removes high-overlap duplicates, but a shadow
    bbox may overlap at only 20–40% IoU — below YOLO's NMS threshold yet still
    an artefact. We apply a tighter post-hoc guard on accepted detections only.

    accepted_boxes : list of (x1, y1, x2, y2) already validated
    new_box        : (x1, y1, x2, y2) candidate box
    iou_threshold  : 0.45 — shadow is typically adjacent, not fully coincident
    """
    nx1, ny1, nx2, ny2 = new_box
    new_area = max(0, nx2 - nx1) * max(0, ny2 - ny1)
    if new_area == 0:
        return True                             # zero-area box — always reject

    for (ax1, ay1, ax2, ay2) in accepted_boxes:
        ix1  = max(nx1, ax1);  iy1 = max(ny1, ay1)
        ix2  = min(nx2, ax2);  iy2 = min(ny2, ay2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            continue
        union = new_area + (ax2 - ax1) * (ay2 - ay1) - inter
        if union > 0 and inter / union > iou_threshold:
            return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — SINGLE IMAGE ANALYSIS
# (Kept from Colab analyze_image() — enhanced with all TRACE features)
# ══════════════════════════════════════════════════════════════════════════════

def analyze_image(img_path, show_summary=True, run_gradcam=False):
    """
    Full 3-phase analysis of a single image.

    Phase 1 : Image Triage         — quality bucket + skip decision
    Phase 2 : Detection + Identity — YOLO → ResNet50 → cosine similarity
    Phase 3 : Classification       — HSV colour morph

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

    # ── Load ──────────────────────────────────────────────────────────────
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

    # ── Phase 2a: Preprocess ──────────────────────────────────────────────
    img_proc = preprocess(img, is_ir=(bucket == "IR_Night"))

    # ── Phase 2b: YOLO detection on preprocessed image ───────────────────
    boxes    = detector(img_proc, verbose=False,
                        conf=CONFIDENCE_THRESHOLD,
                        iou=IOU_THRESHOLD)[0].boxes
    if not len(boxes):
        if show_summary:
            print(f"  EMPTY {img_path.name} — no animal detected")
        return result

    img_area    = img_proc.shape[0] * img_proc.shape[1]
    annotated   = img_proc.copy()
    valid_count = 0

    for i, box in enumerate(boxes):
        conf     = float(box.conf[0])
        cls      = int(box.cls[0])
        x1,y1,x2,y2 = map(int, box.xyxy[0])

        if cls not in TIGER_PROXY_CLASSES: continue
        if conf < CONFIDENCE_THRESHOLD:    continue

        box_ratio = (x2-x1)*(y2-y1) / img_area
        if box_ratio < MIN_BOX_PARTIAL:    continue

        # ── Visibility tier (initial — may be overridden below) ──────────
        visibility = "Full_Body" if box_ratio >= MIN_BOX_FULL else "Partial_Body"

        # ── Padded crop ───────────────────────────────────────────────────
        ih, iw    = img_proc.shape[:2]
        crop      = img_proc[max(0, y1-CROP_PAD) : min(ih, y2+CROP_PAD),
                             max(0, x1-CROP_PAD) : min(iw, x2+CROP_PAD)]
        if crop.size == 0: continue

        # ── Shadow artefact check ─────────────────────────────────────────
        # A tiger's shadow on the ground: near-achromatic (low saturation)
        # and dark (low brightness). Real tiger fur: sat > 40, usually > 80.
        if is_shadow_crop(crop):
            if show_summary:
                print(f"      [SHADOW]  box {i+1} rejected "
                      f"— achromatic+dark crop (shadow artefact)")
            continue

        # ── Corner trace detection ────────────────────────────────────────
        # Partial tiger entering/leaving frame at image borders.
        if is_corner_trace(x1, y1, x2, y2, iw, ih):
            visibility = "Corner_Trace"

        # ── Inter-box overlap guard ───────────────────────────────────────
        # Prevents a shadow bbox adjacent to the real tiger from being
        # enrolled as a second tiger (YOLO NMS may not remove it at ~30% IoU).
        accepted_boxes = [d["bbox"] for d in result["detections"]]
        if check_box_overlap(accepted_boxes, (x1, y1, x2, y2)):
            if show_summary:
                print(f"      [OVERLAP] box {i+1} rejected "
                      f"— overlaps an already-accepted detection")
            continue

        # ── Phase 2c: Individual Identity ────────────────────────────────
        fp         = extract_fingerprint(crop)
        # Partial body AND corner traces must never update the DB
        is_partial = (visibility in ("Partial_Body", "Corner_Trace"))
        tiger_id, match_score, is_new = match_or_enroll(fp, update_db=not is_partial)

        if visibility == "Corner_Trace":
            id_status = ("Possible New — Corner" if is_new else
                         f"Corner Match {match_score:.2f}")
        elif visibility == "Partial_Body":
            id_status = ("Possible New — Partial" if is_new else
                         f"Partial Match {match_score:.2f}")
        else:
            id_status = "New Enrollment" if is_new else "Recaptured Match"

        # ── Phase 3: Colour morph ─────────────────────────────────────────
        morph = classify_color_morph(crop)
        tinfo = TIGER_TYPES.get(morph, {"display": morph, "color": (0,255,0)})

        # ── Grad-CAM (optional) ───────────────────────────────────────────
        if run_gradcam:
            gcam_path = GRADCAM_DIR / f"gradcam_{img_path.stem}_det{i+1}.jpg"
            generate_gradcam(crop, gcam_path, label=f"{tiger_id}|{morph}")

        # ── Draw bounding box on annotated image ──────────────────────────
        # Corner traces get a distinct dashed-looking cyan border (3px + 1px gap)
        color = (255, 255, 0) if visibility == "Corner_Trace" else tinfo["color"]
        thickness = 2 if visibility == "Corner_Trace" else 3
        cv2.rectangle(annotated, (x1,y1), (x2,y2), color, thickness)

        lbl1 = f"{tiger_id}  {tinfo['display']}"
        lbl2 = f"Det:{conf:.0%}  Match:{match_score:.2f}  [{visibility}]"
        (lw,lh),_ = cv2.getTextSize(lbl1, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        bg_top = max(0, y1 - lh*2 - 14)
        cv2.rectangle(annotated, (x1, bg_top), (x1+max(lw+8,240), y1), color, -1)
        tc = (0,0,0) if morph in ("White","Snow_White","Golden") else (255,255,255)
        cv2.putText(annotated, lbl1, (x1+4, y1-lh-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, tc, 2)
        cv2.putText(annotated, lbl2, (x1+4, y1-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, tc, 1)

        valid_count += 1
        result["detections"].append({
            "tiger_id"   : tiger_id,
            "morph"      : morph,
            "display"    : tinfo["display"],
            "visibility" : visibility,
            "det_conf"   : round(conf, 4),
            "match_score": round(match_score, 4),
            "is_new"     : is_new,
            "id_status"  : id_status,
            "bbox"       : (x1, y1, x2, y2),
        })

    result["tiger_count"] = valid_count

    # ── Save annotated image ──────────────────────────────────────────────
    if valid_count > 0:
        cv2.imwrite(str(ANNOTATED_DIR / img_path.name), annotated)

    if show_summary:
        print(f"  {'🐯' if valid_count > 0 else '—'} "
              f"{img_path.name}  |  {valid_count} tiger(s)  |  {bucket}")
        for d in result["detections"]:
            print(f"      {d['tiger_id']:<12} {d['display']:<22} "
                  f"{d['visibility']:<14} match={d['match_score']:.2f}  "
                  f"{d['id_status']}")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — BATCH PROCESSING + POPULATION REPORT
# (Replaces Colab Cell 5.4 — adds unique census, morph breakdown, text report)
# ══════════════════════════════════════════════════════════════════════════════

def run_population_report(data_dir=None, run_gradcam=False):
    """
    Process all images in data_dir and generate the forest population report.
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    images   = sorted([p for p in data_dir.rglob("*")
                       if p.suffix.lower() in SUPPORTED_EXTS])

    if not images:
        print(f"[ERROR] No images found in {data_dir}")
        return

    print("=" * 65)
    print("  TIGER AI SYSTEM — POPULATION REPORT")
    print(f"  Survey: {RANGE_NAME} | {BEAT_NAME}")
    print(f"  Images: {len(images)} | Data: {data_dir}")
    print("=" * 65)

    load_db()

    all_results      = []
    csv_rows         = []
    unusable_count   = 0
    no_tiger_count   = 0
    tiger_img_count  = 0
    total_detections = 0

    for idx, img_path in enumerate(images, 1):
        print(f"[{idx:3d}/{len(images)}]", end=" ")
        res = analyze_image(img_path, show_summary=True,
                            run_gradcam=run_gradcam)
        all_results.append(res)

        if res["skipped"]:
            unusable_count += 1
        elif res["tiger_count"] == 0:
            no_tiger_count += 1
        else:
            tiger_img_count  += 1
            total_detections += res["tiger_count"]

        for d in res["detections"]:
            csv_rows.append({
                "Image_File"       : res["filename"],
                "Range"            : RANGE_NAME,
                "Beat"             : BEAT_NAME,
                "Image_Quality"    : res["quality_bucket"],
                "Tigers_In_Frame"  : res["tiger_count"],
                "Visibility"       : d["visibility"],
                "Tiger_ID"         : d["tiger_id"],
                "Is_New_Tiger"     : d["is_new"],
                "Color_Morph"      : d["morph"],
                "ID_Status"        : d["id_status"],
                "Match_Score"      : d["match_score"],
                "Detection_Conf"   : d["det_conf"],
            })

    # ── Save CSV ──────────────────────────────────────────────────────────
    if csv_rows:
        pd.DataFrame(csv_rows).to_csv(OUTPUT_CSV, index=False)

    save_db()

    # ── Morph breakdown ───────────────────────────────────────────────────
    morph_counts = {}
    for row in csv_rows:
        m = row["Color_Morph"]
        morph_counts[m] = morph_counts.get(m, 0) + 1

    # ── Build text report ─────────────────────────────────────────────────
    ts    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "=" * 60,
        "   FOREST TIGER POPULATION REPORT",
        "=" * 60,
        f"  Generated        : {ts}",
        f"  Survey Location  : {RANGE_NAME} | {BEAT_NAME}",
        f"  Models Used      : YOLOv8n (pretrained) + ResNet50 (pretrained)",
        f"  Identity Method  : Cosine Similarity (threshold {SIMILARITY_THRESHOLD})",
        f"  Artefact Guards  : Shadow filter + Corner-trace label + Overlap dedup",
        "=" * 60,
        "",
        "  IMAGE PROCESSING SUMMARY",
        "  " + "─" * 40,
        f"  Total images processed  : {len(images)}",
        f"  Images with tigers      : {tiger_img_count}",
        f"  Images with no tiger    : {no_tiger_count}",
        f"  Unusable images         : {unusable_count}",
        "",
        "  CENSUS RESULTS",
        "  " + "─" * 40,
        f"  Total detections        : {total_detections}",
        f"  UNIQUE TIGERS (census)  : {len(tiger_db)}",
        "  (same tiger in 10 images = 1 individual, not 10)",
        "",
        "  COLOUR MORPH BREAKDOWN",
        "  " + "─" * 40,
    ]

    for morph, cnt in sorted(morph_counts.items(), key=lambda x: -x[1]):
        pct  = cnt / total_detections * 100 if total_detections else 0
        bar  = "█" * int(pct / 4)
        disp = TIGER_TYPES.get(morph, {}).get("display", morph)
        lines.append(f"  {disp:<24}: {cnt:3d}  ({pct:5.1f}%)  {bar}")

    lines += [
        "",
        "  INDIVIDUAL SIGHTING COUNTS",
        "  " + "─" * 40,
    ]
    for tid, cnt in sorted(tiger_counts.items()):
        lines.append(f"  {tid:<14}: {cnt} sighting(s)")

    lines += [
        "",
        "  OUTPUT FILES",
        "  " + "─" * 40,
        f"  CSV Report       : {OUTPUT_CSV}",
        f"  Identity DB      : {DB_FILE}",
        f"  Annotated images : {ANNOTATED_DIR}",
        f"  Grad-CAM outputs : {GRADCAM_DIR}",
        "",
        "=" * 60,
        "  END OF REPORT",
        "=" * 60,
    ]

    report = "\n".join(lines)
    print("\n\n" + report)
    REPORT_TXT.write_text(report, encoding="utf-8")
    print(f"\n[INFO] Report saved → {REPORT_TXT}")
    print(f"[INFO] CSV saved    → {OUTPUT_CSV}")
    print(f"[INFO] DB saved     → {DB_FILE}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tiger AI System — Local Complete Version")
    parser.add_argument("--image",   type=str, default=None,
                        help="Analyse a single image by filename")
    parser.add_argument("--data",    type=str, default=None,
                        help="Path to image folder (default: data/sample/tigers/train)")
    parser.add_argument("--gradcam", action="store_true",
                        help="Generate Grad-CAM visualisations for each detection")
    args = parser.parse_args()

    if args.image:
        # Single image mode
        load_db()
        img_path = DATA_DIR / args.image
        if not img_path.exists():
            img_path = Path(args.image)
        result = analyze_image(img_path, show_summary=True,
                               run_gradcam=args.gradcam)
        save_db()
    else:
        # Batch mode — full population report
        run_population_report(data_dir=args.data, run_gradcam=args.gradcam)
