# 🐯 AI-Driven Tiger Enumeration — 20-Day Project Plan
## Submission: June 10, 2026 | Start: May 23, 2026

---

## SYNOPSIS

**What we are building:**
A 3-stage Computer Vision pipeline that takes raw camera trap images from the Sundarbans and automatically detects tigers, counts how many are in each frame, and identifies which individual tiger it is — using stripe pattern recognition.

**Why it matters:**
The forest department currently reviews thousands of images manually. Our system replaces that with an AI pipeline that is faster, consistent, and scalable to every tiger reserve in India.

**The 3 outputs:**
1. `Tiger detected: YES / NO` — with confidence score
2. `Count: 2 tigers in this image` — bounding boxes drawn
3. `Identity: T-17, T-23` — individual recognised by stripes

---

## TABLE OF CONTENTS

| # | Section |
|---|---------|
| 1 | Project Architecture |
| 2 | 20-Day Phase Plan |
| 3 | What to Learn (Prerequisites) |
| 4 | Learning Resources |
| 5 | Deliverables Checklist |

---

# PART 1 — PROJECT ARCHITECTURE

## The Full Pipeline (End to End)

```
┌─────────────────────────────────────────────────────────────────┐
│                     INPUT LAYER                                  │
│   Camera Trap Image (JPG/PNG — any size, day or night)          │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PREPROCESSING LAYER                            │
│                                                                  │
│  1. Load image (OpenCV)                                          │
│  2. CLAHE contrast enhancement  ← fixes dark night images       │
│  3. Resize to 224×224                                            │
│  4. Normalise pixels to [0,1]                                    │
│  5. Augmentation (training only) ← flip, blur, brightness       │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                STAGE 1 — DETECTION                               │
│                ResNet50 (Transfer Learning)                      │
│                                                                  │
│  Input:  224×224×3 image                                        │
│  Output: Tiger=1 / No Tiger=0  +  confidence score             │
│  Target: > 90% accuracy, False Negative Rate < 5%              │
│                                                                  │
│  How: ResNet50 pretrained on ImageNet (frozen)                  │
│       + new classification head (Dense → Dropout → Sigmoid)    │
│       Phase 2: unfreeze top 30 layers, fine-tune               │
└─────────────────────────┬───────────────────────────────────────┘
                          │ Tiger confirmed ✅
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                STAGE 2 — COUNTING                                │
│                YOLOv8m (Object Detection)                        │
│                                                                  │
│  Input:  640×640 image                                          │
│  Output: N bounding boxes  +  count  +  each tiger crop        │
│  Target: mAP@0.5 > 0.75, < 2 sec per image                    │
│                                                                  │
│  How: YOLOv8m pretrained weights                                │
│       Fine-tuned on tiger-annotated dataset                     │
│       YOLO format labels (x, y, w, h per tiger)                │
└─────────────────────────┬───────────────────────────────────────┘
                          │ N crops extracted
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                STAGE 3 — INDIVIDUAL ID                           │
│                EfficientNetB3 + Cosine Embeddings                │
│                                                                  │
│  Input:  300×300×3 tiger crop                                   │
│  Output: Tiger ID (T-01, T-17...) + similarity score           │
│  Target: > 80% top-1 accuracy                                  │
│                                                                  │
│  How: EfficientNetB3 pretrained (fine-tune last 20 layers)     │
│       1536-dim embedding vector extracted                        │
│       Cosine similarity against known tiger database            │
│       New tiger? → Register as new individual                   │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     OUTPUT LAYER                                 │
│                                                                  │
│  📊 Per image:   Count=2 | T-17 ✓ | T-23 ✓ | Confidence: 94%  │
│  🗺  Per session: Camera B7 | 2:14AM | Sector 3 | 2 tigers     │
│  📈 Population:  14 unique individuals seen this month          │
│  🚨 Alerts:      Unknown tiger detected — new individual?       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Architecture Diagram — Layers

```
DATA LAYER          PREPROCESSING     MODEL LAYER          OUTPUT
──────────          ─────────────     ───────────          ──────
Camera trap   →     CLAHE         →   ResNet50        →    Tiger Y/N
images              Resize            (Detection)
(raw JPG)           Normalise
                    Augment       →   YOLOv8          →    Count + Boxes
                                      (Counting)

                                  →   EfficientNetB3  →    Individual ID
                                      (Identification)
                                                      →    Embeddings DB
```

---

## Technology Stack

```
LANGUAGE        Python 3.10+
DEEP LEARNING   TensorFlow 2.13 / Keras
DETECTION       Ultralytics YOLOv8
IMAGE OPS       OpenCV, Pillow, Albumentations
DATA            NumPy, Pandas
VISUALISATION   Matplotlib, Seaborn
EXPERIMENT      MLflow
NOTEBOOKS       Jupyter
VERSION CTRL    Git + GitHub
```

---

# PART 2 — 20-DAY PHASE PLAN

## Phase Overview

```
MAY 23 ──────────────────────────────────────────────── JUNE 10
  │                                                         │
  ├── PHASE 0 ──┤ PHASE 1 ────┤ PHASE 2 ───┤ PHASE 3 ──┤ PHASE 4
  Days 1-2      Days 3-6      Days 7-11    Days 12-16   Days 17-20
  Foundation    Data          Models       Evaluation   Submission
```

---

## PHASE 0 — Foundation Setup (Days 1–2 | May 23–24)

**Goal:** Environment ready, team aligned, prerequisites studied.

### Tasks
- [ ] Clone repo: `git clone https://github.com/bhargavimeduri/epaib-batch05-group4`
- [ ] Create virtual environment + install `requirements.txt`
- [ ] Verify all imports work: TensorFlow, OpenCV, Ultralytics, Albumentations
- [ ] All team members add names to README.md
- [ ] Study prerequisite concepts (see Part 3 below)
- [ ] Confirm data delivery timeline with Anamitra / Wildlife Dept

### Deliverable
```
✅ Everyone can run: python -c "import tensorflow; import cv2; import ultralytics"
✅ Team meeting notes updated in docs/meeting_notes.md
✅ Data ETA confirmed
```

---

## PHASE 1 — Data & Pipeline (Days 3–6 | May 25–28)

**Goal:** Understand the dataset, build the full preprocessing pipeline.

### Day 3 — Data Exploration (Notebook 01)
- [ ] Place received images in `data/raw/tiger/` and `data/raw/no_tiger/`
- [ ] Run `notebooks/01_data_exploration.ipynb`
- [ ] Count images per class — are we balanced?
- [ ] Check image sizes, quality, night vs day ratio
- [ ] Show CLAHE enhancement before/after on 5 sample images
- [ ] Document findings in notebook markdown cells

### Day 4 — Preprocessing Validation
- [ ] Run full `preprocess_pipeline()` on all images
- [ ] Check for corrupt files (try-except on every image)
- [ ] Create `data/processed/train/`, `val/`, `test/` splits (70/15/15)
- [ ] Verify no data leakage between splits

### Day 5 — Augmentation Pipeline (Notebook 02)
- [ ] Add Albumentations pipeline to `src/preprocessing.py`
- [ ] Show 9 augmented versions of same tiger image
- [ ] Confirm augmented dataset is 5x original size
- [ ] Visualise HOG (Histogram of Oriented Gradients) on tiger image
  — this makes stripe patterns visible as numbers

### Day 6 — YOLO Annotation
- [ ] Install LabelImg: `pip install labelImg`
- [ ] Annotate 50–100 images with bounding boxes in YOLO format
- [ ] Format: one `.txt` per image, `0 cx cy w h` (normalised 0–1)
- [ ] Run `prepare_yolo_dataset()` to create `data.yaml`

### Deliverable
```
✅ data/processed/ has train/val/test splits
✅ Notebook 01 + 02 complete with visualisations
✅ At least 100 YOLO-annotated images ready
✅ Augmentation pipeline validated
```

---

## PHASE 2 — Model Training (Days 7–11 | May 29 – June 2)

**Goal:** Train all 3 models. Get baseline numbers.

### Day 7 — Stage 1: Detection Model (ResNet50 Phase 1)
- [ ] Create `src/train.py` with `TrainingConfig` dataclass
- [ ] Train ResNet50 — frozen base, 20 epochs
- [ ] Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- [ ] Log to MLflow
- [ ] Save best weights: `models/detection_v1.keras`
- [ ] Target: > 85% validation accuracy by end of day

### Day 8 — Stage 1: Fine-Tuning + Grad-CAM
- [ ] Call `unfreeze_and_finetune()` — unfreeze top 30 layers
- [ ] Train 10 more epochs at lr=1e-5
- [ ] Save: `models/detection_best.keras`
- [ ] **Implement Grad-CAM heatmap** in Notebook 03
  — overlay heatmap on tiger image
  — confirm: model looks at stripes, NOT trees
- [ ] Target: > 90% validation accuracy

### Day 9 — Stage 2: YOLOv8 Object Detection
- [ ] Create `src/yolo_detector.py` with `TigerDetector` class
- [ ] Fine-tune YOLOv8m on tiger YOLO dataset: 50 epochs
- [ ] Save: `models/yolo_tiger.pt`
- [ ] Test on 3 images — verify bounding boxes appear correctly
- [ ] Test on 1 image with 2 tigers — count must = 2

### Day 10 — Stage 3: Individual ID Model (EfficientNetB3)
- [ ] Requires: per-tiger labelled data (T-01, T-02... subfolders)
- [ ] Train EfficientNetB3 — 30 epochs
- [ ] Save: `models/identification_best.keras`
- [ ] Extract 1536-dim embeddings for entire validation set
- [ ] Run t-SNE (2D projection) — clusters should = tiger IDs

### Day 11 — Full Pipeline Integration
- [ ] Connect all 3 stages in Notebook 03 end-to-end
- [ ] Single camera trap image → COUNT + IDs output
- [ ] Measure total pipeline speed (target: < 2 sec/image)
- [ ] Fix any integration bugs

### Deliverable
```
✅ models/detection_best.keras saved
✅ models/yolo_tiger.pt saved
✅ models/identification_best.keras saved
✅ Grad-CAM heatmap shows stripes, not background
✅ t-SNE plot shows natural tiger clusters
✅ End-to-end pipeline runs on one test image
```

---

## PHASE 3 — Evaluation & Insights (Days 12–16 | June 3–7)

**Goal:** Prove the models work. Generate conservation insights.

### Day 12 — Detection Model Evaluation
- [ ] Run Notebook 04 — detection metrics
- [ ] Compute: Accuracy, Precision, Recall, F1, AUC-ROC
- [ ] Plot confusion matrix
- [ ] **Critical:** Check False Negative Rate — must be < 5%
- [ ] Show worst failures — WHY did it miss that tiger?

### Day 13 — YOLOv8 + ID Model Evaluation
- [ ] YOLOv8: compute mAP@0.5
- [ ] Plot Precision-Recall curve
- [ ] Count accuracy: does the model get the count exactly right?
- [ ] Individual ID: Top-1 and Top-5 accuracy
- [ ] Similarity search: query tiger → nearest match from database

### Day 14 — Conservation Insights
- [ ] Tigers per camera trap location (heatmap)
- [ ] Unique individuals identified (population count)
- [ ] Time-of-day activity distribution (when are tigers active?)
- [ ] Frequency per individual (which tigers appear most/least?)
- [ ] Frame these as slides for IIM presentation

### Day 15 — Error Analysis
- [ ] What types of images fail most? (night? occlusion? cubs?)
- [ ] Collect 10 failure examples with explanations
- [ ] Suggest: "With X more images of this type, accuracy would improve"
- [ ] Document limitations honestly — IIM faculty respects this

### Day 16 — Report Writing
- [ ] Fill in `reports/final_report.md`
- [ ] Sections: Abstract, Problem, Data, Architecture, Results, Insights, Limitations, Future Work
- [ ] All figures saved to `reports/figures/`
- [ ] < 10 pages — concise and visual

### Deliverable
```
✅ Notebook 04 complete with all metrics
✅ FNR < 5% (or documented why not achieved)
✅ Conservation insights — at least 4 visualisations
✅ Final report draft complete
```

---

## PHASE 4 — Submission (Days 17–20 | June 7–10)

**Goal:** Polish, present, submit.

### Day 17 — Presentation Build
- [ ] Create slides in `presentation/` folder
- [ ] Slide structure:
  1. Title + team
  2. Problem (the manual pain)
  3. Solution overview (3-stage pipeline diagram)
  4. Data (1TB, govt partnership)
  5. Stage 1 results + Grad-CAM
  6. Stage 2 YOLOv8 + counting demo
  7. Stage 3 ID + t-SNE cluster
  8. Conservation impact numbers
  9. Limitations + future work
  10. Demo (live or recorded)

### Day 18 — Code Cleanup + README Final
- [ ] All notebooks run top-to-bottom without errors
- [ ] Remove any hardcoded paths — use `pathlib`
- [ ] Final README review
- [ ] Update PROJECT_PLAN.md with actual results vs targets

### Day 19 — GitHub Final Push + Submission Package
- [ ] Push all code, notebooks, report
- [ ] Tag the final commit: `git tag v1.0-submission`
- [ ] Create submission ZIP if required
- [ ] Final check: does `git clone` + `pip install -r requirements.txt` work clean?

### Day 20 — Buffer / Submission June 10
- [ ] Submit to Dr. Sowmya S
- [ ] Share GitHub link with Wildlife Department
- [ ] Celebrate 🐯

---

# PART 3 — WHAT TO LEARN (Prerequisites)

## Learn These Before You Start Coding

---

### CONCEPT 1 — How Images Become Data (Day 1)
**What to understand:**
> An image is a numpy array. A 224×224 colour image = array of shape (224, 224, 3).
> Each value = 0 to 255. This is your raw "unstructured data."

**Practice:**
```python
import cv2
import numpy as np

img = cv2.imread("tiger.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(img.shape)       # (height, width, channels)
print(img[0][0])       # first pixel: [R, G, B]
print(img.max())       # 255
img_norm = img / 255.0 # normalise
```

**Why it matters:** Everything in this project starts here. You can't train a model on raw pixels — you need to understand what you're feeding it.

---

### CONCEPT 2 — CNNs: How the Model Sees (Day 1-2)
**What to understand:**
> A CNN scans the image with small filters (3×3).
> Each filter detects one feature: edge, colour gradient, curve.
> Deeper layers combine features: edge + curve = stripe.
> Final layers: stripe + orange = tiger.

**Key terms to know:**
- `Convolution` — sliding filter across image
- `Feature map` — the output after a filter scans
- `Pooling` — shrinking the feature map (keep important, discard noise)
- `Flatten` — turn 2D feature maps into 1D vector
- `Dense layer` — fully connected, makes final decision

**Watch:** 3Blue1Brown "But what is a neural network?" (YouTube, 20 min)

---

### CONCEPT 3 — Transfer Learning (Day 2)
**What to understand:**
> ResNet50 was trained on 1.2 million images.
> It already knows edges, textures, shapes, fur patterns.
> You take that knowledge, add YOUR classification head, train only the head.
> This is transfer learning — stand on giants' shoulders.

**Two modes:**
```
Feature Extraction:  base.trainable = False  ← fast, less data needed
Fine-Tuning:         unfreeze top layers     ← slow, more accurate
```

**Why ResNet50 specifically:**
> Residual connections (skip connections) prevent vanishing gradients.
> 50 layers deep — learns complex patterns.
> Proven on wildlife imagery in research papers.

---

### CONCEPT 4 — Object Detection vs Classification (Day 2)
**What to understand:**
```
CLASSIFICATION          DETECTION
"Is there a tiger?"     "Where is each tiger? How many?"
One output              N outputs (one per object)
No location             Bounding box (x, y, width, height)
```

**YOLO specifically:**
> Divides image into grid.
> Each grid cell predicts: box coordinates + confidence + class.
> Single forward pass = all detections at once. FAST.
> YOLOv8 is the latest version — best accuracy + speed.

**Key terms:**
- `Bounding box` — rectangle around each tiger
- `IoU (Intersection over Union)` — how well does predicted box overlap ground truth?
- `mAP (mean Average Precision)` — main YOLO accuracy metric
- `Confidence threshold` — only show boxes above X% confidence
- `NMS (Non-Maximum Suppression)` — remove duplicate boxes

---

### CONCEPT 5 — Embeddings and Similarity (Day 3)
**What to understand:**
> After training, cut the model before the final softmax layer.
> What you get is a vector of numbers (e.g., 1536 numbers).
> This vector is the model's compressed representation of the tiger.
> Two images of the SAME tiger → similar vectors (close in space).
> Two DIFFERENT tigers → different vectors (far in space).

**Cosine Similarity:**
```python
from sklearn.metrics.pairwise import cosine_similarity
sim = cosine_similarity(embedding1, embedding2)
# 1.0 = identical, 0.0 = completely different
```

**Why this is powerful:**
> You don't need to have seen a tiger before to match it.
> "Is this T-17?" → compute similarity → if > 0.85, it's T-17.
> New tiger? → No match above threshold → register as new individual.

---

### CONCEPT 6 — Evaluation Metrics for Images (Day 3)
**What you must understand before you evaluate:**

| Metric | Formula | Use when |
|--------|---------|---------|
| Accuracy | Correct / Total | Balanced dataset |
| Precision | TP / (TP + FP) | False positives are costly |
| Recall | TP / (TP + FN) | **False negatives are costly ← our case** |
| F1 | 2×P×R / (P+R) | Balance both |
| AUC-ROC | Area under ROC curve | Overall discrimination ability |
| mAP@0.5 | Mean AP at IoU=0.5 | Object detection (YOLO) |

**For tigers: Recall is most important.**
> Missing a tiger (False Negative) is worse than a false alarm.
> Target: False Negative Rate < 5% per problem statement.

---

### CONCEPT 7 — Data Augmentation (Day 4)
**What to understand:**
> Small dataset + big model = overfitting (memorises, doesn't generalise).
> Augmentation = artificially create more training images.
> Each augmentation is a realistic variation the model might see in the wild.

**Realistic for camera traps:**
```python
HorizontalFlip      ← tiger can face either direction
BrightnessContrast  ← day vs dusk vs night
Rotate(±15°)        ← camera angle variation
MotionBlur          ← tiger is moving
CoarseDropout       ← simulates foliage blocking the view
```

**NOT realistic (avoid):**
```
VerticalFlip   ← tigers don't walk upside down
Extreme colour ← don't destroy the orange-black signature
```

---

### CONCEPT 8 — Grad-CAM (Day 7-8)
**What to understand:**
> Grad-CAM answers: "WHY did the model say tiger?"
> It computes gradients of the prediction with respect to the last conv layer.
> Produces a heatmap — bright = "this region caused the decision."

**Expected result:** Heatmap should glow over the tiger's body (stripes, face) — NOT over trees or sky. If it glows over background → model is cheating, not learning tigers.

**Why it matters for IIM presentation:**
> This is visual proof your model learned the RIGHT thing.
> One slide with Grad-CAM = 10x more convincing than accuracy numbers alone.

---

# PART 4 — LEARNING RESOURCES

## Free Resources (Watch/Read in This Order)

| # | Resource | Time | Topic |
|---|----------|------|-------|
| 1 | 3Blue1Brown — Neural Networks (YouTube playlist) | 1 hr | CNNs fundamentals |
| 2 | CS231n Stanford — Lecture 5 (YouTube) | 1 hr | CNNs for images |
| 3 | Keras Transfer Learning tutorial (keras.io) | 30 min | ResNet50 in code |
| 4 | Ultralytics YOLOv8 Docs — Train section | 30 min | YOLOv8 training |
| 5 | Albumentations official docs | 20 min | Augmentation |
| 6 | StatQuest — ROC and AUC (YouTube) | 15 min | Evaluation metrics |

## Hands-On Practice Before Day 7

```python
# Mini-exercise 1: Load and preprocess one image
import cv2
img = cv2.imread("any_image.jpg")
img = cv2.resize(img, (224, 224))
img = img / 255.0
print("Shape:", img.shape)  # should be (224, 224, 3)

# Mini-exercise 2: Load ResNet50, print summary
from tensorflow.keras.applications import ResNet50
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
model.summary()  # read the layer names — understand what each does

# Mini-exercise 3: Run YOLOv8 on any image
from ultralytics import YOLO
model = YOLO('yolov8m.pt')
results = model('any_image.jpg')
results[0].show()  # see bounding boxes
```

---

# PART 5 — DELIVERABLES CHECKLIST

## What to Submit on June 10

```
GITHUB REPO
├── ✅ README.md (visual, badges, pipeline diagram)
├── ✅ context/ (official govt letters)
├── ✅ docs/problem_statement.md
├── ✅ docs/architecture.md
├── ✅ src/preprocessing.py (with augmentation)
├── ✅ src/model.py (with embeddings)
├── ✅ src/train.py
├── ✅ src/yolo_detector.py
├── ✅ notebooks/01_data_exploration.ipynb (run, outputs visible)
├── ✅ notebooks/02_preprocessing_augmentation.ipynb (run)
├── ✅ notebooks/03_model_training.ipynb (Grad-CAM visible)
├── ✅ notebooks/04_evaluation_insights.ipynb (metrics + conservation)
├── ✅ reports/final_report.md
└── ✅ presentation/ (slides)

METRICS TO REPORT
├── Detection accuracy: target > 90%
├── False Negative Rate: target < 5%
├── YOLOv8 mAP@0.5: target > 0.75
├── Individual ID accuracy: target > 80%
├── Pipeline speed: target < 2 sec/image
└── Unique tigers identified: actual count from dataset

CONSERVATION OUTPUTS
├── Total unique tigers in dataset
├── Tigers per camera trap (heatmap)
├── Activity time distribution
└── Population estimate
```

---

## Day-by-Day Summary

| Day | Date | Phase | Key Task |
|-----|------|-------|---------|
| 1-2 | May 23-24 | Setup | Environment + prerequisites study |
| 3 | May 25 | Data | Explore dataset, Notebook 01 |
| 4 | May 26 | Data | Preprocessing, train/val/test split |
| 5 | May 27 | Data | Augmentation pipeline, Notebook 02 |
| 6 | May 28 | Data | YOLO annotation (LabelImg) |
| 7 | May 29 | Model | ResNet50 Phase 1 training |
| 8 | May 30 | Model | Fine-tune + Grad-CAM |
| 9 | May 31 | Model | YOLOv8 training |
| 10 | Jun 1 | Model | EfficientNetB3 ID training |
| 11 | Jun 2 | Model | Full pipeline integration |
| 12 | Jun 3 | Eval | Detection metrics |
| 13 | Jun 4 | Eval | YOLOv8 + ID metrics |
| 14 | Jun 5 | Eval | Conservation insights |
| 15 | Jun 6 | Eval | Error analysis |
| 16 | Jun 7 | Report | Final report writing |
| 17 | Jun 8 | Submit | Presentation slides |
| 18 | Jun 8 | Submit | Code cleanup |
| 19 | Jun 9 | Submit | Final GitHub push + tag |
| 20 | Jun 10 | 🎯 | SUBMIT |

---

*Last updated: 2026-05-23 | Group 4 | EPAIB Batch 05 | IIM Lucknow*
*Faculty: Dr. Sowmya S | sowmya@iiml.ac.in*
