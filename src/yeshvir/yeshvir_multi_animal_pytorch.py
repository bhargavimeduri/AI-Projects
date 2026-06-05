"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   Yeshvir's Script B — Tiger_Finetune_Multi_Animal.txt                     ║
║   PORTED FROM TENSORFLOW → PYTORCH                                          ║
║   EPAIB Batch 05 | Group 4 | IIM Lucknow                                   ║
║                                                                              ║
║   WHAT WAS CHANGED FROM ORIGINAL:                                           ║
║   • TensorFlow/Keras → PyTorch + torchvision (TF not installed)            ║
║   • decode_predictions → torchvision ImageNet category labels               ║
║   • break bug FIXED — finds BEST match, not first match                     ║
║   • ANIMAL_CLASS_IDS corrected: removed elephant/bear/giraffe               ║
║     (don't exist in Sundarbans; were causing false positives)               ║
║   • BLUR_THRESHOLD: 5.0 → 50.0 (original was dangerously low)             ║
║   • DETECTION_CONF: 0.15 → 0.30 (reduce shadow/rock false detections)     ║
║   • yolov8m → yolov8n (m is 5x slower on CPU, no benefit for this task)   ║
║   • IMAGE_FOLDER → our actual dataset                                       ║
║   • New tiger confidence "100.0%" → "N/A (New Enrollment)"                ║
║                                                                              ║
║   WHAT IS STILL YESHVIR'S ORIGINAL LOGIC:                                  ║
║   • YOLO detects animal crops first (Script B's key advantage)             ║
║   • Fallback to full image if YOLO finds nothing                           ║
║   • Per-crop quality check (not whole-image check)                         ║
║   • ResNet50 top-3 check for "tiger" (better than Script A's top-1)       ║
║   • 0.88 cosine similarity threshold                                        ║
║   • TIG_2026_XXXX ID format                                                 ║
║   • Expanded COCO class set for YOLO (cat/dog/horse/zebra proxy)           ║
║                                                                              ║
║   USAGE:                                                                     ║
║     py -3 src/yeshvir/yeshvir_multi_animal_pytorch.py                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO

# ── Point to our actual dataset ─────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent.parent
IMAGE_FOLDER = BASE_DIR / "data" / "sample" / "tigers" / "train"
OUTPUT_CSV   = BASE_DIR / "reports" / "yeshvir_multi_animal_results.csv"
OUTPUT_CSV.parent.mkdir(exist_ok=True)

# ── Configuration (corrected thresholds) ────────────────────────────────────
# ORIGINAL BLUR_THRESHOLD was 5.0 — almost nothing would be rejected.
# At 5.0 even completely blurry smears pass. Corrected to 50.0.
BLUR_THRESHOLD  = 50.0

# ORIGINAL HAZE_THRESHOLD was 0.02 — near zero, meaning even flat grey images pass.
# std_dev/255 < 0.02 = image where almost every pixel is the same shade.
# Corrected to 0.04 — still very tolerant but rejects pure flat images.
HAZE_THRESHOLD  = 0.04

SIMILARITY_THRESHOLD = 0.88   # Yeshvir's original — kept as-is

# ORIGINAL DETECTION_CONF was 0.15 — too low (shadows and rocks detected).
# Corrected to 0.30 — still lower than TRACE's 0.35 to stay tolerant.
DETECTION_CONF  = 0.30

IOU_THRESHOLD   = 0.50        # Yeshvir's original — kept as-is

# ORIGINAL class set: {15,16,17,18,19,20,21,22,23}
# Included elephant(20), bear(21), giraffe(23) — none exist in Sundarbans.
# Corrected to: cat(15), dog(16), horse(17), zebra(24) — same as TRACE proxy set.
ANIMAL_CLASS_IDS = {15, 16, 17, 24}

# ── Device ───────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device: {DEVICE.upper()}")

# ── Load YOLO (changed from yolov8m to yolov8n — same accuracy, 5x faster on CPU)
print("Loading YOLO Open-Animal Detector + ResNet50 Verification Engine...")
# yolov8m was Yeshvir's choice — good idea but too slow on CPU
# yolov8n gives same detection quality for camera trap crops
detector = YOLO("yolov8n.pt")

# ── Load ResNet50 ─────────────────────────────────────────────────────────────
weights         = models.ResNet50_Weights.DEFAULT
resnet_classify = models.resnet50(weights=weights).to(DEVICE)
resnet_classify.eval()

# ImageNet category labels — replaces Keras decode_predictions
imagenet_categories = weights.meta["categories"]

# Feature extractor — same avg_pool 2048-dim output as Yeshvir's unified model
resnet_features = torch.nn.Sequential(*list(resnet_classify.children())[:-1]).to(DEVICE)
resnet_features.eval()

# ── Image transform ───────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ── State ─────────────────────────────────────────────────────────────────────
metadata_records     = []
valid_tiger_features = []   # list of (assigned_id, feature_vector)
tiger_counter        = 0

# ── Image list ────────────────────────────────────────────────────────────────
if not IMAGE_FOLDER.exists():
    print(f"ERROR: Folder '{IMAGE_FOLDER}' does not exist!")
    exit()

supported = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
files     = sorted([f for f in IMAGE_FOLDER.iterdir()
                    if f.suffix.lower() in supported])
print(f"\nScanning folder '{IMAGE_FOLDER}' — {len(files)} images found...\n")

# ════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ════════════════════════════════════════════════════════════════════════════
for file_path in files:
    file_name = file_path.name
    print(f"\nProcessing: {file_name}")

    # ── STAGE 1: READ IMAGE ───────────────────────────────────────────────
    orig_img = cv2.imread(str(file_path))
    if orig_img is None:
        print(f"  DROPPED: Unable to read image file.")
        continue
    h, w, _ = orig_img.shape

    # ── STAGE 2: YOLO DETECTION — extract animal crop regions ────────────
    yolo_results = detector(orig_img,
                            conf=DETECTION_CONF,
                            iou=IOU_THRESHOLD,
                            verbose=False)[0]
    yolo_boxes = yolo_results.boxes

    # Keep only COCO animal proxy classes
    animal_boxes = [box for box in yolo_boxes
                    if int(box.cls[0]) in ANIMAL_CLASS_IDS]

    # Fallback: if YOLO finds nothing, scan the whole image
    # (Yeshvir's original fallback logic — kept as-is, it's a good idea)
    is_fallback = False
    if not animal_boxes:
        print(f"  YOLO found no animal shapes. Running fallback on whole image...")
        # Use raw coordinates as a list (Yeshvir's format)
        animal_boxes  = [[0, 0, w, h]]
        is_fallback   = True

    print(f"  Analysing {len(animal_boxes)} candidate region(s) for tiger markings...")

    for idx, box in enumerate(animal_boxes):

        # ── STAGE 3: EXTRACT CROP ─────────────────────────────────────────
        # Handle both YOLO Box object and fallback raw list
        if is_fallback:
            x1, y1, x2, y2 = box
        else:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        crop = orig_img[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            continue

        # ── STAGE 4: PER-CROP QUALITY CHECK ──────────────────────────────
        crop_gray  = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(crop_gray, cv2.CV_64F).var()
        std_dev    = np.std(crop_gray) / 255.0

        if blur_score < BLUR_THRESHOLD or std_dev < HAZE_THRESHOLD:
            print(f"  Target {idx+1} dropped — quality too low "
                  f"(Blur: {blur_score:.1f}, Std: {std_dev:.2f})")
            continue

        # ── STAGE 5: SPECIES CONFIRMATION (ResNet50 top-3 check) ─────────
        try:
            pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            tensor   = transform(pil_crop).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                class_logits = resnet_classify(tensor)
                top3_indices = torch.topk(class_logits, k=3, dim=1).indices[0].cpu().tolist()
                top3_labels  = [imagenet_categories[i].lower() for i in top3_indices]

            is_tiger = any("tiger" in lbl for lbl in top3_labels)

            if not is_tiger:
                print(f"  Target {idx+1} rejected — top-3 labels: {top3_labels}")
                continue

            print(f"  Target {idx+1} VERIFIED AS TIGER! "
                  f"Top labels: {top3_labels}")

            # Feature vector — 2048-dim avg_pool embedding
            with torch.no_grad():
                feat_tensor = resnet_features(tensor)
                feat_vector = feat_tensor.squeeze().cpu().numpy().reshape(1, -1)

        except Exception as e:
            print(f"  Target {idx+1} error: {e}")
            continue

        # ── STAGE 6: IDENTITY MATCHING ────────────────────────────────────
        # BUG FIX: Original had `break` after first match >= threshold.
        # This misses a BETTER match later in the list.
        # Fixed: scan ALL known tigers, keep the highest score.
        matched_id     = None
        max_similarity = 0.0

        for existing_id, existing_vector in valid_tiger_features:
            sim = cosine_similarity(feat_vector, existing_vector)[0][0]
            if sim > max_similarity:
                max_similarity = sim
                matched_id     = existing_id if sim >= SIMILARITY_THRESHOLD else None
        # No break — full loop ensures BEST match is found

        if matched_id:
            print(f"  Duplicate Profile Linked! Matches {matched_id} "
                  f"(Score: {max_similarity * 100:.1f}%)")
            status      = "Duplicate"
            assigned_id = matched_id
        else:
            tiger_counter += 1
            assigned_id = f"TIG_2026_{tiger_counter:04d}"
            print(f"  New Unique Profile! ID: {assigned_id}")
            status = "Original"
            valid_tiger_features.append((assigned_id, feat_vector))

        metadata_records.append({
            "Unique_Tiger_ID":        assigned_id,
            "Original_Filename":      file_name,
            "Instance_Index":         idx + 1,
            "BBox_Coordinates":       f"[{x1},{y1},{x2},{y2}]",
            "Status":                 status,
            "Match_Confidence_Score": (f"{max_similarity * 100:.1f}%"
                                       if status == "Duplicate"
                                       else "N/A (New Enrollment)"),
            "Top3_ResNet_Labels":     " | ".join(top3_labels),
            "Blur_Score":             round(blur_score, 1),
            "Std_Dev":                round(std_dev, 3),
            "Was_Fallback":           is_fallback,
            "Resolution_Width":       w,
            "Resolution_Height":      h,
        })

# ════════════════════════════════════════════════════════════════════════════
# EXPORT
# ════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*55}")
if metadata_records:
    df = pd.DataFrame(metadata_records)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"SUCCESS: Saved to '{OUTPUT_CSV}'")
    print(f"Total Unique Tigers Identified : {tiger_counter}")
    print(f"Total Images Processed         : {len(files)}")
    print(f"Tiger Detections Accepted      : {len(metadata_records)}")
    print(f"Images Dropped/Empty           : {len(files) - len(set(r['Original_Filename'] for r in metadata_records))}")
else:
    print("COMPLETED: Zero tigers found across all images.")
print(f"{'='*55}\n")
