# Learning Notes — Colab Script vs TRACE Pipeline
## EPAIB Batch 05 | Group 4 | IIM Lucknow
### Topic: What We Built, What Was Missing, and Why It Matters

---

## 1. The Two Scripts — What Each Was Trying to Do

### Original Colab Script (`copy_of_tiger_ai_system_complete.py`)
- A **Google Colab notebook** exported as `.py`
- Goal: Train a YOLO detector + EfficientNetB3 classifier, then run inference
- What it does well: Bounding box visualisation, clean reporting, single-image analysis function
- What it could not do: Run locally (hard Colab dependencies), individual tiger identity, census counting

### TRACE Pipeline (`tiger_pipeline.py` → `tiger_ai_complete.py`)
- A **local Python pipeline** built for the real use case
- Goal: Detect, identify, and count unique individual tigers (census)
- What it does well: Identity tracking, DB persistence, preprocessing, reproducibility
- What it was missing (before v5): Bounding box visualisation, crop padding, visibility labels

---

## 2. Full Comparison Table

| Feature | Colab Script | TRACE (v4) | Fixed Script (`tiger_ai_complete.py`) |
|---|---|---|---|
| **Runs locally** | No — Colab only | Yes | Yes |
| **Needs GPU** | Yes — crashes without it | No | No (uses CPU fallback) |
| **Needs training** | Yes — 90 min training first | No | No |
| **Individual tiger ID** | Missing | Tiger_001, Tiger_002… | Tiger_001, Tiger_002… |
| **Census (unique count)** | Missing — counts detections | 13 unique tigers | 13 unique tigers |
| **Identity DB persistence** | Missing | JSON save/load | JSON save/load |
| **Running average embedding** | Missing | Yes — DB improves per sighting | Yes |
| **CLAHE preprocessing** | Basic brightness only | Yes — LAB L-channel | Yes |
| **Dark Channel Prior dehazing** | Missing | Yes — fog/rain/haze | Yes |
| **IR/night detection** | Converts mode only | Yes — channel std analysis | Yes |
| **Image triage with skip** | Quality scan only | Yes — blur/over/under | Yes |
| **Partial body DB protection** | Missing | Yes — `update_db=False` | Yes |
| **Grad-CAM explainability** | Missing | Separate script | Built in (`--gradcam` flag) |
| **Bounding box visualisation** | Yes — colour-coded | Missing | Yes — colour-coded |
| **Crop padding** | 10px pad | Missing | 15px pad |
| **Visibility label** | full/partial/trace | Not in CSV | Full_Body / Partial_Body |
| **YOLO on preprocessed image** | Passes original path (bug) | Bug — same | Fixed — passes array |
| **Morph classification** | EfficientNetB3 (needs training) | HSV rule-based | HSV rule-based |
| **Single-image analysis** | `analyze_image()` function | Not exposed | `analyze_image()` function |
| **Population report** | Yes | Yes (console only) | Yes (text file + CSV) |

---

## 3. Why the Colab Script Could NOT Identify 2 Tigers in One Image

**Three root causes:**

### Cause 1 — Fine-tuned YOLO trained on single-tiger images (most likely)
When you fine-tune YOLO on a dataset where most frames contain one tiger,
the model learns a "one high-confidence box per image" pattern.
It missed the second tiger because it never learned to find two simultaneously.

**Fix:** Retrain on a dataset with multi-tiger annotations. Or use pretrained
COCO YOLOv8n (our approach) — COCO has multi-instance images by default.

### Cause 2 — Confidence threshold filtering the second tiger
```python
conf_threshold = 0.15   # Cell 5.2 in Colab
```
If tiger 1 has confidence 0.42 and tiger 2 has confidence 0.12,
tiger 2 is below 0.15 and gets filtered out before you even see it.
**Fix:** Lower threshold to 0.08 and inspect what boxes appear.

### Cause 3 — NMS IoU threshold merging overlapping boxes
When two tigers are close together in frame, their bounding boxes overlap.
```python
iou_threshold = 0.7   # Cell 5.2 in Colab
```
NMS (Non-Maximum Suppression) removes boxes that overlap by more than the IoU threshold.
At 0.7, boxes need 70%+ overlap to be suppressed — this is actually permissive.
But at 0.3, boxes with only 30% overlap get suppressed — the second tiger disappears.
**Fix:** Keep IoU at 0.5–0.7 for multi-tiger images.

---

## 4. Key Technical Concepts Learned

### 4.1 Transfer Learning — Two Ways to Use It
| Approach | What you do | When to use |
|---|---|---|
| Feature extraction | Remove classification head, use 2048-dim vector | No labelled identity data (our case) |
| Fine-tuning | Keep head, train on your labels with low LR | When you have labelled training data |

**Our situation:** We have NO tiger identity ground truth labels.
So we use ResNet50 as a feature extractor — the 2048-dim vector
captures visual similarity WITHOUT needing training labels.

### 4.2 Cosine Similarity vs Euclidean Distance
| Metric | What it measures | When to use |
|---|---|---|
| Cosine similarity | Angle between two vectors (direction) | When scale varies — day/night, near/far |
| Euclidean distance | Absolute distance between two points | When scale is consistent |

**Why cosine for tiger identity:**
- Same tiger photographed in daylight vs infrared night = different brightness levels
- Euclidean distance would say they're far apart (different brightness = different scale)
- Cosine ignores brightness scale, compares only the PATTERN of the embedding
- The stripe pattern stays consistent even when brightness changes

### 4.3 CLAHE — Why Not Standard Histogram Equalisation?
Standard HE: Stretches the histogram of the whole image.
Problem: If one tiger is dark and the background is bright, HE brightens
the background more and makes the tiger worse.

CLAHE (Contrast Limited Adaptive HE):
- Divides image into 8×8 tiles
- Equalises each tile independently (local, not global)
- `clipLimit=2.0` — prevents over-amplification of noise
- Applied ONLY to the L channel in LAB colour space (not RGB channels directly)
- Result: Tiger body gets enhanced without blowing out background

```python
lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
l_enhanced = clahe_processor.apply(l)   # only L, not a or b
```

### 4.4 Dark Channel Prior Dehazing — How It Works
Published by He et al. (IEEE 2010) — one of the most cited CV papers.

**The observation:** In most non-sky patches of a clear outdoor image,
at least one colour channel has very low intensity (near zero).
Haze/fog raises this minimum — a "foggy" patch has higher dark channel.

**The algorithm:**
1. Compute dark channel: minimum pixel value across R/G/B channels in a local patch
2. Estimate atmospheric light A: from the top 0.1% brightest dark-channel pixels
3. Compute transmission map t: `t = 1 - ω × J_dark / A` (ω=0.95 removes 95% of haze)
4. Recover scene: `J = (I - A) / max(t, t₀) + A`

**Why it matters for camera traps:**
Sundarbans monsoon season = heavy fog, rain scatter.
Without dehazing, YOLO struggles to find tigers behind the haze.

### 4.5 Running Average Embedding — Why Not Just Replace?
When you re-sight Tiger_003 for the 5th time, you have two choices:
- **Replace:** DB = new embedding from this sighting only
- **Running average:** DB = weighted average of all 5 sightings

```python
new_embedding = (old_embedding × n + new_embedding) / (n + 1)
```

**Why running average is better:**
- A single sighting may be partially occluded, poorly lit, or low resolution
- Averaging across 5 sightings smooths out noise
- The DB embedding becomes a more reliable fingerprint over time
- Each new sighting IMPROVES the model rather than replacing it

### 4.6 Why Partial Body Crops Must Not Update the DB
If Tiger_003's tail is visible in one image:
- The 2048-dim embedding of a tiger tail is NOT the same as a full-body embedding
- If you use it to update the DB, you corrupt the running average
- Next time Tiger_003 is seen fully, the cosine similarity drops below 0.83
- Tiger_003 gets re-enrolled as Tiger_014 — false new tiger count

```python
match_or_enroll(fingerprint, update_db=False)  # match only, never modify
```

### 4.7 Grad-CAM — What It Proves
Grad-CAM (Gradient-weighted Class Activation Mapping):
1. Forward pass → get class scores
2. Select the top predicted class
3. Backward pass → compute gradients of that class score w.r.t. last conv layer features
4. Global average pool the gradients → importance weight per channel
5. Weighted sum of feature maps → 7×7 activation map
6. ReLU + normalise + resize to original crop size

**What it tells us:**
- RED on the heatmap = model focused here for this prediction
- BLUE = model ignored this area
- Good: RED on tiger body/stripes
- Bad: RED on background trees or corner of the image

**Quote from Prof. Mahesh Balan (IIT Madras PhD, Walmart):**
> "A model you can't explain is a model you can't trust."

In insurance claims fraud (same principle): a fraud model that flags a claim
must show WHY it flagged it. Regulators require explanation, not just a score.
Same here — a tiger pipeline that says "new tiger" must be able to show
the features it used to make that decision.

---

## 5. What the Fixed Script Added Over Both

`tiger_ai_complete.py` = TRACE features + Colab structure

**New features in the fixed version:**
1. `analyze_image()` function — clean single-image API (from Colab)
2. Bounding box visualisation with morph-specific colours (from Colab)
3. Crop padding (15px) — prevents edge artefacts (from Colab)
4. Visibility label in CSV (Full_Body / Partial_Body)
5. Grad-CAM integrated as `--gradcam` flag (no separate script needed)
6. YOLO receives preprocessed `img` array (not original file path)
7. `Tigers_In_Frame` counts only valid detections (not all YOLO boxes)
8. Text report saved to `reports/tiger_population_report.txt`

---

## 6. How to Run the Fixed Script

```bash
cd "C:\Users\91630\OneDrive\Desktop\Bhargavi AIB\epaib-batch05-group4"

# Full batch — all 187 train images
py -3 src/tiger_ai_complete.py

# Single image
py -3 src/tiger_ai_complete.py --image tiger_10.png

# Full batch with Grad-CAM on every detection
py -3 src/tiger_ai_complete.py --gradcam

# Custom data folder
py -3 src/tiger_ai_complete.py --data "data/sample/tigers/val"
```

**Output files:**
```
reports/
├── tiger_population_report.csv    ← per-detection CSV
├── tiger_population_report.txt    ← human-readable summary
├── tiger_registry_db.json         ← persistent identity DB
├── annotated_images/              ← bounding box images
│   ├── tiger_10.png               ← each image with coloured boxes
│   └── ...
└── gradcam_outputs/               ← Grad-CAM (if --gradcam used)
    ├── gradcam_tiger_10_det1.jpg
    └── ...
```

---

## 7. Open Questions (for Prof. Sowmya's Review)

| Question | Why It Matters |
|---|---|
| Threshold 0.83 — how was it chosen? | Need ROC curve on labelled val images to calibrate |
| 13 unique tigers from 187 images — is this right? | Need ground truth to verify |
| Subspecies classifier is a placeholder | Do not present subspecies output as a result |
| Domain shift — Google Images vs real camera traps | Embeddings may not generalise |

---

*Last updated: 2026-05-31 | Group 4 | EPAIB Batch 05 | IIM Lucknow*
