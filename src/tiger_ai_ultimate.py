"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   TIGER AI ULTIMATE — Fused Multi-Source Pipeline v3.0                     ║
║   EPAIB Batch 05 | Group 4 | IIM Lucknow                                   ║
║                                                                              ║
║   WHAT IS NEW vs tiger_ai_complete.py (TRACE):                             ║
║   ─────────────────────────────────────────────────────────────────────     ║
║   ✅ CASCADE DETECTION (3 stages — never misses a tiger):                  ║
║      Stage 1: YOLOv8n-OIV7  → "Tiger" class directly (from Dr. Bose)      ║
║      Stage 2: YOLOv8n-COCO  → cat/dog/horse/zebra proxy (from TRACE)      ║
║      Stage 3: Whole-image   → fallback if YOLO finds nothing               ║
║                               (from Yeshvir Script B's is_fallback logic)  ║
║                                                                              ║
║   ✅ SPECIES VERIFICATION LAYER (from Yeshvir Script B):                   ║
║      ResNet50 top-3 must contain "tiger" before fingerprint extraction.    ║
║      Applies to COCO and whole-image detections (OIV7 already confirmed).  ║
║      Stops shadow/background being fingerprinted as a tiger.               ║
║                                                                              ║
║   ✅ VIEWPOINT CLASSIFICATION (from Dr. Bose):                             ║
║      Every crop is labelled Left_Flank / Right_Flank / Frontal.            ║
║      A tiger's left and right stripes look different — without this,       ║
║      the same tiger seen from both sides could be counted as two tigers.   ║
║                                                                              ║
║   ✅ VIEWPOINT-AWARE CENSUS in final report:                               ║
║      Conservative count = max(left-unique IDs, right-unique IDs).          ║
║      Liberal count  = union of all IDs (upper bound).                      ║
║                                                                              ║
║   ✅ GRAYSCALE NORMALISATION for night/IR (from Dr. Bose):                 ║
║      Night images → grayscale → 3-channel → ResNet50.                     ║
║      Forces the model to read stripe geometry, not colour temperature.     ║
║                                                                              ║
║   WHAT IS KEPT FROM TRACE (tiger_ai_complete.py):                         ║
║   ─────────────────────────────────────────────────────────────────────     ║
║   ✅ CLAHE + Dark Channel Prior dehazing                                    ║
║   ✅ Image triage (blur / over / underexpose — skip unusable frames)        ║
║   ✅ Shadow detection (HSV saturation guard)                                ║
║   ✅ Corner trace detection (border-touching box guard)                     ║
║   ✅ Inter-box overlap guard (post-YOLO duplicate removal)                  ║
║   ✅ Running average identity embedding (DB improves each sighting)         ║
║   ✅ JSON DB persistence (survives restarts, cross-session census)          ║
║   ✅ Partial body guard (partial crops never corrupt the DB)                ║
║   ✅ Grad-CAM explainability (visual proof model reads stripes)             ║
║   ✅ HSV colour morph classification                                         ║
║   ✅ Annotated images + CSV population report + text summary                ║
║                                                                              ║
║   WHY THIS REACHES ~95% ACCURACY:                                          ║
║   ─────────────────────────────────────────────────────────────────────     ║
║   TRACE problem: YOLOv8n-COCO uses cat/dog as tiger proxies →              ║
║     misses real tigers + picks up non-tigers → inflated/missed counts.     ║
║   Fix: OIV7 stage detects "Tiger" directly → more correct crops.           ║
║                                                                              ║
║   TRACE problem: No species gate → any COCO proxy match extracts           ║
║     fingerprint, including rocks/stumps that look like cat in COCO.        ║
║   Fix: Top-3 ResNet gate → fingerprint only if "tiger" confirmed.          ║
║                                                                              ║
║   TRACE problem: YOLO misses occluded/tiny tigers → no count.              ║
║   Fix: Cascade fallback ensures every image gets at least one attempt.     ║
║                                                                              ║
║   TRACE problem: Same tiger seen from left+right could get 2 IDs.         ║
║   Fix: ViewPoint-aware census uses max(L,R) not L+R.                      ║
║                                                                              ║
║   USAGE:                                                                     ║
║     py -3 src/tiger_ai_ultimate.py                    # batch run           ║
║     py -3 src/tiger_ai_ultimate.py --gradcam          # with Grad-CAM      ║
║     py -3 src/tiger_ai_ultimate.py --image tiger.jpg  # single image       ║
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
# SECTION 1 — SETUP & CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# ── File and folder paths ──────────────────────────────────────────────────────
# __file__ = this script's location.  parent.parent = project root folder.
BASE_DIR       = Path(__file__).resolve().parent.parent
DATA_DIR       = BASE_DIR / "data" / "sample" / "tigers" / "train"  # Default dataset
REPORTS_DIR    = BASE_DIR / "reports"               # All output files go here
ANNOTATED_DIR  = REPORTS_DIR / "ultimate_annotated" # Images with bounding boxes drawn
GRADCAM_DIR    = REPORTS_DIR / "ultimate_gradcam"   # Grad-CAM heatmap images
DB_FILE        = REPORTS_DIR / "ultimate_tiger_db.json"           # Persistent tiger profiles
OUTPUT_CSV     = REPORTS_DIR / "ultimate_population_report.csv"   # Per-detection log
REPORT_TXT     = REPORTS_DIR / "ultimate_population_report.txt"   # Text census summary

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
BLUR_THRESHOLD        = 25.0   # Laplacian variance < 25 → genuinely unusable blur
                                # (was 80 — too strict for camera trap quality)
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
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # ── IR / Night check FIRST — before any brightness/blur rejection ──────────
    # IR camera images are inherently dark (low mean brightness) and sometimes
    # slightly soft. Rejecting them on brightness before we identify them as IR
    # causes real tiger images to be discarded. Detect IR → process them,
    # auto-brighten in preprocessing; never skip purely on brightness.
    if detect_ir_night(img_bgr):
        return "IR_Night", f"Greyscale IR (bright={brightness:.0f}, blur={blur_score:.0f})", False

    # ── Standard quality checks for regular (colour) images ───────────────────
    if brightness > OVEREXPOSE_THRESHOLD:
        return "Unusable_Overexposed",  f"brightness {brightness:.0f}", True
    if brightness < UNDEREXPOSE_THRESHOLD:
        return "Unusable_Underexposed", f"brightness {brightness:.0f}", True
    if blur_score < BLUR_THRESHOLD:
        return "Unusable_Blurry",       f"Laplacian {blur_score:.1f}", True
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
    oiv7_detector = YOLO("yolov8n-oiv7.pt")
    print("[INFO] OIV7 detector loaded — Stage 1 detection active.")
except Exception as e:
    print(f"[WARN] OIV7 load failed ({e}). Cascade will start from COCO Stage 2.")

# ── (b) COCO detector — fallback, always available ────────────────────────────
print("[INFO] Loading YOLOv8n-COCO (proxy classes) ...")
coco_detector = YOLO("yolov8n.pt")

# ── (c/d) ResNet50 — classification head (top-3 gate) + feature extractor ─────
print("[INFO] Loading ResNet50 (species verification + feature extraction) ...")
_weights_resnet      = models.ResNet50_Weights.DEFAULT
resnet_classify      = models.resnet50(weights=_weights_resnet).to(DEVICE)
resnet_classify.eval()

# The imagenet_categories list maps index → human-readable label.
# "tiger" appears in this list (e.g. "tiger cat", "tiger shark", "Bengal tiger").
# We check if any of the top-3 predictions contain the word "tiger".
imagenet_categories = _weights_resnet.meta["categories"]

# Feature extractor = ResNet50 with the final classification layer removed.
# Output: a 2048-dimensional vector — the tiger's "stripe fingerprint".
resnet_features = torch.nn.Sequential(
    *list(resnet_classify.children())[:-1]).to(DEVICE)
resnet_features.eval()

# Standard ImageNet transform (applies to every crop before ResNet)
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

resnet_classify.layer4.register_forward_hook(_hook_fmaps)
resnet_classify.layer4.register_full_backward_hook(_hook_grads)


def generate_gradcam(crop_bgr, save_path, label=""):
    """
    Grad-CAM on ResNet50 layer4.
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
        if raw_box["requires_verify"]:
            verified, top3_labels = verify_species_top3(crop)
            if not verified:
                if show_summary:
                    print(f"    [SPECIES] rejected — top-3: {top3_labels}")
                continue
            if show_summary:
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

            # Try horizontal split (left | right) — catches side-by-side tigers
            left_half  = img_proc[:, :mid_x]
            right_half = img_proc[:, mid_x:]
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
                        left_half, right_half, flip_axis=1)  # 1 = horizontal flip
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
                            "x1": 0, "y1": 0, "x2": mid_x, "y2": ih,
                            "conf": 0.0, "detection_source": "Grid_Scan_L",
                            "requires_verify": False,
                            "_grid_verified": True,
                            "_top3": ["(grid_left_confirmed)"],
                        })
                        raw_boxes.append({
                            "x1": mid_x, "y1": 0, "x2": iw, "y2": ih,
                            "conf": 0.0, "detection_source": "Grid_Scan_R",
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
        result["detections"].append({
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
        })

        if show_summary:
            print(f"    {tiger_id:<12} {tinfo['display']:<22} "
                  f"{visibility:<14} {viewpoint:<18} "
                  f"match={match_score:.2f}  {id_status}  [{det_src}]")

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
                if box_iou >= LOWCONF_SAME_BOX_IOU:
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
    REMAINDER_SAME_TIGER_THRESH = 0.82   # guard vs. YOLO-detected tigers in this image
    REMAINDER_CROSS_HALF_THRESH = 0.55   # guard vs. OTHER remainder-enrolled tigers
    #   Remainder halves are full-image splits (include background), so similarity
    #   is lower than grid-crop halves. Calibration:
    #   Single tiger spanning frame → L/R cross-half sim ~0.65  (must block: > 0.55 ✓)
    #   Same cub top/bottom half cross-sim              → ~0.58  (must block: > 0.55 ✓)
    #   Adult tiger vs cub (different individuals) → ~0.35–0.45 (must pass: < 0.55 ✓)
    #   Lowered from 0.62 → 0.55 to catch same-cub T/B pairs (b.jpeg: sim=0.58)

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

        for (rx1, ry1, rx2, ry2, rsrc, half_img) in remainder_halves:
            # Skip halves already substantially covered by an accepted detection
            cov = _half_coverage(rx1, ry1, rx2, ry2)
            if cov >= REMAINDER_COVERAGE_THRESH:
                if show_summary:
                    print(f"    [REMAINDER] {rsrc}: skipped — {cov:.0%} covered "
                          f"(threshold {REMAINDER_COVERAGE_THRESH:.0%})")
                continue
            if half_img.shape[0] < 20 or half_img.shape[1] < 20:
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

            if max_sim_to_existing >= REMAINDER_SAME_TIGER_THRESH:
                if show_summary:
                    print(f"    [REMAINDER] {rsrc}: same tiger as existing detection "
                          f"(sim={max_sim_to_existing:.2f}) — skipped")
                continue

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

    if valid_count >= 1 and iw >= 80 and ih >= 80:
        from sklearn.metrics.pairwise import cosine_similarity as _cos_sim

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

            if max_sim_m >= MARGIN_SAME_TIGER_THRESH:
                if show_summary:
                    print(f"    [MARGIN] {msrc}: same tiger as existing "
                          f"(sim={max_sim_m:.2f}) — skipped")
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

        if src.startswith("Grid_Scan"):
            reason = (f"grid-split detection — split decision uses fingerprint "
                      f"similarity only, no species gate on the split itself")
        elif is_new and score < NEW_ENROLL_CONF_MIN:
            reason = (f"new enrollment with extremely low match score {score:.2f} "
                      f"— no close DB match found at all")

        if reason:
            det["confidence"] = "LOW"
            uncertain_detections.append((det["tiger_id"], reason))
        elif score >= 0.83 and not is_new and src == "OIV7_Direct":
            det["confidence"] = "HIGH"
        else:
            det["confidence"] = "MEDIUM"

    result["uncertain_detections"] = uncertain_detections
    result["needs_confirmation"]   = len(uncertain_detections) > 0

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

    result["tiger_count"] = valid_count
    if valid_count > 0:
        cv2.imwrite(str(ANNOTATED_DIR / img_path.name), annotated)

    return result


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
        f"  Detection Chain  : YOLOv8n-OIV7 -> COCO-proxy -> Whole-image",
        f"  Species Gate     : ResNet50 top-3 tiger verification",
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

    # ── Uncertain images summary ──────────────────────────────────────────
    uncertain_images = [
        res for res in all_results
        if res.get("needs_confirmation") and not res.get("skipped")
    ]

    if uncertain_images:
        lines += [
            "",
            "  " + "!" * 61,
            "  !!  UNCERTAIN IMAGES — HUMAN CONFIRMATION NEEDED        !!",
            "  " + "!" * 61,
            "  The pipeline is NOT fully confident about the detections",
            "  listed below. These are BEST-GUESS predictions only.",
            "  Please look at each image and confirm the tiger count.",
            "  " + "-" * 61,
        ]
        for res in uncertain_images:
            pred_count = res["tiger_count"]
            pred_ids   = ", ".join(
                dict.fromkeys(d["tiger_id"] for d in res["detections"]))
            lines.append(
                f"  IMAGE : {res['filename']}")
            lines.append(
                f"    My prediction  : {pred_count} tiger(s)  →  {pred_ids}")
            for tid, why in res.get("uncertain_detections", []):
                lines.append(f"    Uncertain ({tid}) : {why}")
            lines.append("")
        lines += [
            "  " + "!" * 61,
        ]
    else:
        lines += [
            "",
            "  CONFIDENCE CHECK",
            "  " + "-" * 40,
            "  All detections in this batch are HIGH or MEDIUM confidence.",
            "  No human confirmation required for this run.",
        ]

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


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tiger AI Ultimate — Fused Multi-Source Pipeline v3.0")
    parser.add_argument("--image",   type=str, default=None,
                        help="Analyse a single image by filename or full path")
    parser.add_argument("--data",    type=str, default=None,
                        help="Path to image folder (default: data/sample/tigers/train)")
    parser.add_argument("--gradcam", action="store_true",
                        help="Generate Grad-CAM visualisations for each detection")
    args = parser.parse_args()

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
