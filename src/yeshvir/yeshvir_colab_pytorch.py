"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   Yeshvir's Script A — Tiger_Finetune_Colab.txt                            ║
║   PORTED FROM TENSORFLOW → PYTORCH                                          ║
║   EPAIB Batch 05 | Group 4 | IIM Lucknow                                   ║
║                                                                              ║
║   WHAT WAS CHANGED FROM ORIGINAL:                                           ║
║   • TensorFlow/Keras → PyTorch + torchvision (TF not installed)            ║
║   • decode_predictions → torchvision ImageNet category labels               ║
║   • break bug FIXED — now finds BEST match, not first match                 ║
║   • IMAGE_FOLDER → points to our actual dataset                             ║
║   • New tiger confidence "100.0%" → "N/A (New Enrollment)" (was misleading)║
║                                                                              ║
║   WHAT IS STILL YESHVIR'S ORIGINAL LOGIC:                                  ║
║   • No YOLO — whole image fed to ResNet50 (Script A's approach)            ║
║   • Quality check on whole image (blur + std_dev)                          ║
║   • ResNet50 top-1 must contain "tiger" to accept the image                ║
║   • 0.88 cosine similarity threshold                                        ║
║   • TIG_2026_XXXX ID format                                                 ║
║                                                                              ║
║   USAGE:                                                                     ║
║     py -3 src/yeshvir/yeshvir_colab_pytorch.py                             ║
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

# ── Point to our actual dataset ─────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent.parent
IMAGE_FOLDER = BASE_DIR / "data" / "sample" / "tigers" / "train"
OUTPUT_CSV   = BASE_DIR / "reports" / "yeshvir_colab_results.csv"
OUTPUT_CSV.parent.mkdir(exist_ok=True)

# ── Yeshvir's original thresholds (kept as-is) ──────────────────────────────
BLUR_THRESHOLD       = 30.0
HAZE_THRESHOLD       = 0.05
SIMILARITY_THRESHOLD = 0.88

# ── Device ───────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device: {DEVICE.upper()}")

# ── Load ResNet50 — classification model (with head) ─────────────────────────
print("Loading ResNet50 Intelligence & Feature Extraction Layers...")

weights         = models.ResNet50_Weights.DEFAULT
resnet_classify = models.resnet50(weights=weights).to(DEVICE)
resnet_classify.eval()

# ImageNet category labels — lets us find "tiger" in predictions
# (replaces Keras decode_predictions)
imagenet_categories = weights.meta["categories"]   # list of 1000 label strings

# ── Feature extractor (head removed — avg_pool output = 2048-dim vector) ─────
# This is Yeshvir's feature_model = Model(inputs, outputs=avg_pool)
resnet_features = torch.nn.Sequential(*list(resnet_classify.children())[:-1]).to(DEVICE)
resnet_features.eval()

# ── Image transform (standard ImageNet preprocessing) ────────────────────────
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ── State ────────────────────────────────────────────────────────────────────
metadata_records   = []
valid_tiger_features = []   # list of (assigned_id, feature_vector)
tiger_counter      = 0

# ── Image list ───────────────────────────────────────────────────────────────
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
    print(f"Processing: {file_name}")

    # ── STAGE 1: READ & QUALITY CHECK ─────────────────────────────────────
    orig_img = cv2.imread(str(file_path))
    if orig_img is None:
        print(f"  DROPPED: Unable to read image.")
        continue

    h, w, _ = orig_img.shape
    gray     = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)

    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    std_dev    = np.std(gray) / 255.0

    if blur_score < BLUR_THRESHOLD or std_dev < HAZE_THRESHOLD:
        print(f"  DROPPED: Failed quality checks "
              f"(Blur: {blur_score:.1f}, Std: {std_dev:.2f})")
        continue

    # ── STAGE 2: FEATURE EXTRACTION + CLASSIFICATION ──────────────────────
    try:
        # Convert BGR (OpenCV) → RGB → PIL → tensor
        pil_img = Image.fromarray(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
        tensor  = transform(pil_img).unsqueeze(0).to(DEVICE)

        # Classification — top-1 label check (Yeshvir's original logic)
        with torch.no_grad():
            class_logits = resnet_classify(tensor)
            top_idx      = int(torch.argmax(class_logits, dim=1).item())
            top_label    = imagenet_categories[top_idx].lower()

        if "tiger" not in top_label:
            print(f"  DROPPED: Subject identified as '{top_label}', not a tiger.")
            continue

        print(f"  VERIFIED AS TIGER — top label: '{top_label}'")

        # Feature vector — 2048-dim from avg_pool (Yeshvir's feature_model)
        with torch.no_grad():
            feat_tensor = resnet_features(tensor)
            feat_vector = feat_tensor.squeeze().cpu().numpy().reshape(1, -1)

    except Exception as e:
        print(f"  Error extracting features: {e}")
        continue

    # ── STAGE 3: IDENTITY MATCHING ────────────────────────────────────────
    # BUG FIX from original: do NOT break early.
    # Original code broke on first match >= threshold.
    # Fixed: loop through ALL known tigers and find the BEST match.
    matched_id     = None
    max_similarity = 0.0

    for existing_id, existing_vector in valid_tiger_features:
        sim = cosine_similarity(feat_vector, existing_vector)[0][0]
        if sim > max_similarity:
            max_similarity = sim
            matched_id     = existing_id if sim >= SIMILARITY_THRESHOLD else None
    # No break — full loop ensures best match is found

    if matched_id:
        print(f"  DUPLICATE! Matches {matched_id} "
              f"(Score: {max_similarity * 100:.1f}%)")
        status      = "Duplicate"
        assigned_id = matched_id
    else:
        tiger_counter += 1
        assigned_id = f"TIG_2026_{tiger_counter:04d}"
        print(f"  NEW UNIQUE TIGER — ID: {assigned_id}")
        status = "Original"
        valid_tiger_features.append((assigned_id, feat_vector))

    metadata_records.append({
        "Unique_Tiger_ID":        assigned_id,
        "Original_Filename":      file_name,
        "Status":                 status,
        "Match_Confidence_Score": (f"{max_similarity * 100:.1f}%"
                                   if status == "Duplicate"
                                   else "N/A (New Enrollment)"),
        "Top_ResNet_Label":       top_label,
        "Blur_Score":             round(blur_score, 1),
        "Std_Dev":                round(std_dev, 3),
        "Resolution_Width":       w,
        "Resolution_Height":      h,
    })

# ════════════════════════════════════════════════════════════════════════════
# EXPORT
# ════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*50}")
if metadata_records:
    df = pd.DataFrame(metadata_records)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"SUCCESS: Saved to '{OUTPUT_CSV}'")
    print(f"Total Unique Tigers Registered : {tiger_counter}")
    print(f"Total Images Processed         : {len(files)}")
    print(f"Tiger Images Accepted          : {len(metadata_records)}")
    print(f"Images Dropped (quality/label) : {len(files) - len(metadata_records)}")
else:
    print("COMPLETED: Zero tigers found across all images.")
print(f"{'='*50}\n")
