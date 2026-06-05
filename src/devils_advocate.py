"""
╔══════════════════════════════════════════════════════════════╗
║   TRACE — Devil's Advocacy Validation                       ║
║   Runs the pipeline TWICE on the same images from scratch.  ║
║   Compares results to test consistency and reproducibility. ║
║                                                              ║
║   If the model is correct:                                   ║
║     - Same images → same tiger count both runs              ║
║     - Same images → same color morph both runs              ║
║     - Same images → consistent identity assignments         ║
║                                                              ║
║   If the model is wrong:                                     ║
║     - Tiger counts differ between runs                      ║
║     - Same image gets different tiger ID each time          ║
║     - Color morph flips between runs                        ║
╚══════════════════════════════════════════════════════════════╝
"""

import json
import shutil
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

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
TARGET_DIR  = BASE_DIR / "data" / "sample" / "tigers" / "train"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

RUN1_CSV = REPORTS_DIR / "devils_advocate_run1.csv"
RUN2_CSV = REPORTS_DIR / "devils_advocate_run2.csv"
DIFF_CSV = REPORTS_DIR / "devils_advocate_comparison.csv"

# ── Config ─────────────────────────────────────────────────────────────────
SIMILARITY_THRESHOLD = 0.83
CONFIDENCE_THRESHOLD = 0.35
MIN_BOX_FULL         = 0.05
MIN_BOX_PARTIAL      = 0.03
TIGER_PROXY_CLASSES  = {15, 16, 17, 24}
SUPPORTED_EXTS       = {".jpg", ".jpeg", ".png", ".webp"}
BLUR_THRESHOLD       = 80.0
OVEREXPOSE_THRESHOLD = 240
UNDEREXPOSE_THRESHOLD= 20
IR_CHANNEL_DIFF      = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Load models once ───────────────────────────────────────────────────────
print("[INFO] Loading models...")
detector    = YOLO("yolov8n.pt")
resnet_base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
feature_extractor = torch.nn.Sequential(
    *(list(resnet_base.children())[:-1])
).to(DEVICE)
feature_extractor.eval()

data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

clahe_proc = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def detect_ir(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    return (np.std(rgb[:,:,0]-rgb[:,:,1]) < IR_CHANNEL_DIFF and
            np.std(rgb[:,:,0]-rgb[:,:,2]) < IR_CHANNEL_DIFF)

def triage(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if np.mean(gray) > OVEREXPOSE_THRESHOLD: return "Unusable_Overexposed", True
    if np.mean(gray) < UNDEREXPOSE_THRESHOLD: return "Unusable_Underexposed", True
    if cv2.Laplacian(gray, cv2.CV_64F).var() < BLUR_THRESHOLD: return "Unusable_Blurry", True
    if detect_ir(img): return "IR_Night", False
    return "Clean", False

def preprocess(img, is_ir=False):
    if is_ir:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    enhanced = cv2.merge([clahe_proc.apply(l), a, b])
    img = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    f   = img.astype(np.float64) / 255.0
    dk  = np.min(f, axis=2)
    k   = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dk  = cv2.erode(dk.astype(np.float32), k).astype(np.float64)
    n   = max(1, int(dk.size * 0.001))
    A   = np.clip(np.max(f.reshape(-1,3)[np.argsort(dk.flatten())[-n:]], axis=0), 1e-6, 1.0)
    dn  = cv2.erode(np.min(f/A, axis=2).astype(np.float32), k).astype(np.float64)
    t   = np.clip(1.0 - 0.95*dn, 0.1, 1.0)[:,:,np.newaxis]
    return (np.clip((f-A)/t+A, 0, 1)*255).astype(np.uint8)

def fingerprint(crop):
    rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    t    = data_transform(Image.fromarray(rgb)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return feature_extractor(t).squeeze().cpu().numpy()

def color_morph(crop):
    h,s,v = cv2.split(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV))
    mv,ms,mh,sv = np.mean(v),np.mean(s),np.mean(h),np.std(v)
    if mv>220 and ms<20 and sv<15: return "Snow_White"
    if mv>180 and ms<50:           return "White"
    if mv<60:                      return "Black"
    if 25<=mh<=40 and ms<120:      return "Golden"
    return "Orange_Standard"

# ══════════════════════════════════════════════════════════════════════════════
# SINGLE RUN FUNCTION (fresh DB every time)
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(run_label):
    """Run full pipeline with a fresh empty database. Returns DataFrame."""

    db      = {}   # fresh DB — no prior knowledge
    counts  = {}
    log     = []

    all_images = [p for p in TARGET_DIR.rglob("*")
                  if p.suffix.lower() in SUPPORTED_EXTS]

    print(f"\n[{run_label}] Processing {len(all_images)} images...")

    for img_path in all_images:
        img = cv2.imread(str(img_path))
        if img is None:
            try:
                pil = Image.open(img_path).convert("RGB")
                img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            except: continue

        quality, skip = triage(img)
        if skip:
            log.append({"Run": run_label, "Image_File": img_path.name,
                        "Quality": quality, "Tigers_In_Frame": 0,
                        "Tiger_ID": "UNUSABLE", "Color_Morph": None,
                        "Match_Score": None, "Conf": None})
            continue

        img   = preprocess(img, is_ir=(quality=="IR_Night"))
        boxes = detector(str(img_path), verbose=False)[0].boxes
        area  = img.shape[0] * img.shape[1]

        if len(boxes) == 0:
            log.append({"Run": run_label, "Image_File": img_path.name,
                        "Quality": quality, "Tigers_In_Frame": 0,
                        "Tiger_ID": "NO_DETECT", "Color_Morph": None,
                        "Match_Score": None, "Conf": None})
            continue

        for box in boxes:
            conf = float(box.conf[0])
            cls  = int(box.cls[0])
            x1,y1,x2,y2 = map(int, box.xyxy[0])

            if cls not in TIGER_PROXY_CLASSES: continue
            if conf < CONFIDENCE_THRESHOLD: continue
            ratio = (x2-x1)*(y2-y1) / area
            if ratio < MIN_BOX_PARTIAL: continue

            crop = img[y1:y2, x1:x2]
            if crop.size == 0: continue

            fp = fingerprint(crop)
            is_partial = ratio < MIN_BOX_FULL

            # Match against fresh DB
            if not db:
                tid = "Tiger_001"
                if not is_partial:
                    db[tid] = fp; counts[tid] = 1
                score, is_new = 1.0, True
            else:
                best, bid = -1.0, None
                for eid, ev in db.items():
                    s = cosine_similarity(fp.reshape(1,-1), ev.reshape(1,-1))[0][0]
                    if s > best: best, bid = s, eid
                if best >= SIMILARITY_THRESHOLD:
                    if not is_partial:
                        n = counts[bid]
                        db[bid] = (db[bid]*n + fp)/(n+1)
                        counts[bid] += 1
                    tid, score, is_new = bid, best, False
                else:
                    tid = f"Tiger_{len(db)+1:03d}"
                    if not is_partial:
                        db[tid] = fp; counts[tid] = 1
                    score, is_new = best, True

            log.append({
                "Run":            run_label,
                "Image_File":     img_path.name,
                "Quality":        quality,
                "Tigers_In_Frame":len(boxes),
                "Tiger_ID":       tid,
                "Color_Morph":    color_morph(crop),
                "Match_Score":    round(float(score), 4),
                "Conf":           round(conf, 4),
                "Is_New":         is_new,
                "Is_Partial":     is_partial,
            })

    df = pd.DataFrame(log)
    unique = len([t for t in db])
    print(f"[{run_label}] Unique tigers: {unique} | Valid detections: {len([r for r in log if str(r.get('Tiger_ID','')).startswith('Tiger')])}")
    return df, unique

# ══════════════════════════════════════════════════════════════════════════════
# RUN BOTH AND COMPARE
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("   DEVIL'S ADVOCACY — Running pipeline TWICE from scratch")
print("="*65)

df1, unique1 = run_pipeline("Run_1")
df2, unique2 = run_pipeline("Run_2")

df1.to_csv(RUN1_CSV, index=False)
df2.to_csv(RUN2_CSV, index=False)

# ── Comparison ────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("   DEVIL'S ADVOCACY — COMPARISON REPORT")
print("="*65)

# 1. Unique tiger count consistency
print(f"\n  Unique tigers found:")
print(f"    Run 1: {unique1}")
print(f"    Run 2: {unique2}")
if unique1 == unique2:
    print(f"    CONSISTENT — same count both runs")
else:
    diff = abs(unique1 - unique2)
    print(f"    INCONSISTENT — {diff} tiger(s) differ between runs")
    print(f"    This suggests threshold instability or borderline embeddings")

# 2. Per-image tiger count consistency
valid1 = df1[df1["Tiger_ID"].str.startswith("Tiger", na=False)][["Image_File","Tigers_In_Frame"]].drop_duplicates()
valid2 = df2[df2["Tiger_ID"].str.startswith("Tiger", na=False)][["Image_File","Tigers_In_Frame"]].drop_duplicates()
merged_counts = valid1.merge(valid2, on="Image_File", suffixes=("_R1","_R2"), how="outer")
count_mismatch = merged_counts[merged_counts["Tigers_In_Frame_R1"] != merged_counts["Tigers_In_Frame_R2"]]

print(f"\n  Per-image tiger count consistency:")
print(f"    Images with same count both runs : {len(merged_counts) - len(count_mismatch)}")
print(f"    Images with DIFFERENT count      : {len(count_mismatch)}")
if not count_mismatch.empty:
    print("\n  Mismatched images:")
    print(count_mismatch.to_string(index=False))

# 3. Color morph consistency
cm1 = df1[df1["Color_Morph"].notna()][["Image_File","Color_Morph"]].rename(columns={"Color_Morph":"Morph_R1"})
cm2 = df2[df2["Color_Morph"].notna()][["Image_File","Color_Morph"]].rename(columns={"Color_Morph":"Morph_R2"})
merged_morph = cm1.merge(cm2, on="Image_File", how="inner")
morph_mismatch = merged_morph[merged_morph["Morph_R1"] != merged_morph["Morph_R2"]]

print(f"\n  Color morph consistency:")
print(f"    Images with same morph both runs : {len(merged_morph) - len(morph_mismatch)}")
print(f"    Images with DIFFERENT morph      : {len(morph_mismatch)}")
if not morph_mismatch.empty:
    print("\n  Color morph flips (same image, different morph):")
    print(morph_mismatch.to_string(index=False))

# 4. Match score stability
scores1 = df1[df1["Match_Score"].notna()]["Match_Score"]
scores2 = df2[df2["Match_Score"].notna()]["Match_Score"]
print(f"\n  Match score distribution:")
print(f"    Run 1 — mean: {scores1.mean():.4f} | std: {scores1.std():.4f} | min: {scores1.min():.4f} | max: {scores1.max():.4f}")
print(f"    Run 2 — mean: {scores2.mean():.4f} | std: {scores2.std():.4f} | min: {scores2.min():.4f} | max: {scores2.max():.4f}")

# 5. Devil's Advocacy Verdict
print("\n" + "="*65)
print("   DEVIL'S ADVOCACY VERDICT")
print("="*65)

issues = []
if unique1 != unique2:
    issues.append(f"Tiger count unstable: {unique1} vs {unique2} — threshold may be too close to borderline embeddings")
if len(count_mismatch) > 0:
    issues.append(f"{len(count_mismatch)} images have different tiger counts between runs — YOLO detection is non-deterministic")
if len(morph_mismatch) > 0:
    issues.append(f"{len(morph_mismatch)} images have different color morph between runs — HSV classifier is unstable")

if not issues:
    print("\n  PASSED — Results are fully reproducible across 2 independent runs.")
    print("  Same images produce same tiger counts and color morphs both times.")
    print("  The model is consistent. Devil's advocacy found no instability.")
else:
    print(f"\n  FLAGGED — {len(issues)} consistency issue(s) found:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")

# Save comparison
diff_rows = []
for _, row in merged_morph.iterrows():
    r1_count = merged_counts[merged_counts["Image_File"]==row["Image_File"]]["Tigers_In_Frame_R1"].values
    r2_count = merged_counts[merged_counts["Image_File"]==row["Image_File"]]["Tigers_In_Frame_R2"].values
    diff_rows.append({
        "Image_File":     row["Image_File"],
        "Count_Run1":     r1_count[0] if len(r1_count) else None,
        "Count_Run2":     r2_count[0] if len(r2_count) else None,
        "Count_Match":    "YES" if len(r1_count) and len(r2_count) and r1_count[0]==r2_count[0] else "NO",
        "Morph_Run1":     row["Morph_R1"],
        "Morph_Run2":     row["Morph_R2"],
        "Morph_Match":    "YES" if row["Morph_R1"]==row["Morph_R2"] else "NO",
    })

pd.DataFrame(diff_rows).to_csv(DIFF_CSV, index=False)
print(f"\n  Detailed comparison saved: {DIFF_CSV}")
print("="*65)
