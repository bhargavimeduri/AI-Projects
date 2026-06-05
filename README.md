<div align="center">

# TRACE — Tiger Recognition and Census Engine
## Tiger AI Ultimate v3.0 | EPAIB Batch 05 | Group 4 | IIM Lucknow

![Tiger](https://img.shields.io/badge/Project-Tiger%20Enumeration-orange?style=for-the-badge)
![IIM](https://img.shields.io/badge/IIM%20Lucknow-EPAIB%20Batch%2005-darkblue?style=for-the-badge)
![Status](https://img.shields.io/badge/Pipeline-v3.0%20Fused-brightgreen?style=for-the-badge)
![Dataset](https://img.shields.io/badge/Validated%20On-ATRW%203392%20Images-yellow?style=for-the-badge)

> **"Count individual tigers, not sightings — automatically, across any dataset, at scale."**

**Group 4 | EPAIB Batch 05 | IIM Lucknow**

</div>

---

## The Problem

The **Sundarbans Tiger Reserve** — world's largest mangrove forest — and tiger reserves across India use **camera traps** to monitor tiger populations. Each camera trap generates thousands of images every week. Every image is currently reviewed manually by forest rangers to:
- Check if a tiger is in the frame
- Identify which individual tiger it is (by stripe pattern)
- Count how many unique tigers were seen this week

```
Manual process today:
  Weeks to review one batch of images
  Rangers diverted from field patrols
  Human fatigue causes missed detections
  Cannot scale to national coverage
```

---

## Our Solution — TRACE Pipeline

A 3-stage cascade AI pipeline that automates the entire workflow:

```
Camera Trap Image
      |
      v
[IMAGE TRIAGE]          Skip blurry, overexposed, underexposed images
      |
      v
[PREPROCESSING]         CLAHE contrast boost + Dark Channel Prior dehazing
                        + IR night normalisation
      |
      v
[CASCADE DETECTION]
  Stage 1: YOLOv8n-OIV7  -->  "Tiger" class directly (Open Images V7)
  Stage 2: YOLOv8n-COCO  -->  Proxy classes (cat/dog/zebra) if Stage 1 misses
  Stage 3: Whole Image   -->  Full-frame scan if both YOLOs find nothing
      |
      v
[SPECIES GATE]          ResNet50 top-3 must confirm "tiger" before accepting
                        (Stops cage bars, rocks, stumps being called tigers)
      |
      v
[FINGERPRINTING]        ResNet50 layer4 -> 2048-dim feature vector
                        (the tiger's unique "stripe fingerprint")
      |
      v
[IDENTITY MATCHING]     Cosine similarity against tiger database
                        Score >= 0.83 -> same tiger, update profile
                        Score < 0.83  -> new tiger, enroll in DB
                        Running average embedding improves with each sighting
      |
      v
[VIEWPOINT CENSUS]      Left_Flank / Right_Flank / Frontal classification
                        Conservative count = max(L, R, F) -- avoids double-counting
      |
      v
OUTPUTS:
  - Annotated images with bounding boxes + tiger IDs
  - Population report CSV (per detection)
  - JSON identity database (persists across runs)
  - Interactive HTML dashboard
  - Word run report
  - Grad-CAM explainability images
```

---

## Key Results

| Dataset | Images | Unique Tigers | Blur Rejection | Mean Match Score |
|---|---|---|---|---|
| Sundarbans (our primary) | 187 | 13 | ~3% | ~0.817 |
| ATRW Amur Tiger (ICCV benchmark) | 3,392 | 11 (at 35%) | 25.9% | 0.796 |
| Test Bh (35 images, new acquisition) | 35 | 7 new + 1 recapture | 14% | 0.89 (recaptures) |

**Key validation:** Pipeline ran on Amur Tiger dataset (completely different species, zoo setting) with **zero code changes** — proving dataset-agnostic design.

---

## Why Each Model Was Chosen

| Model | Role | Why This Model |
|---|---|---|
| YOLOv8n + OIV7 weights | Tiger detection (Stage 1) | Open Images V7 has a direct "Tiger" class — no proxy needed |
| YOLOv8n + COCO weights | Backup detection (Stage 2) | Standard fallback; cat/dog/zebra are shape proxies |
| ResNet50 (classification head) | Species verification gate | Pretrained on ImageNet which includes tiger classes; fast, proven |
| ResNet50 (features only) | 2048-dim stripe fingerprint | Layer4 features encode texture/stripe patterns, not just category |
| Cosine Similarity | Identity matching | Scale-invariant; handles brightness changes across day/night images |
| CLAHE | Contrast enhancement | Recovers tiger stripe texture in low-light / shadow images |
| Dark Channel Prior | Dehazing | Removes Sundarbans monsoon haze that washes out stripe detail |
| Grad-CAM | Explainability | Visual proof the model looks at stripes, not background |

---

## Quick Start

### Prerequisites
- Python 3.9 or above
- Windows / Linux / Mac

### Install

```bash
# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Training (run once before inference on a new dataset)

```bash
# Stage 1 — Fine-tune YOLOv8 on your labeled tiger images
py -3 src/tiger_ai_ultimate.py --train --stage yolo --data-yaml data/tiger_dataset.yaml

# Stage 2 — Train ResNet50 Tiger/No-Tiger binary classifier
py -3 src/tiger_ai_ultimate.py --train --stage detection --train-data data/processed

# Stage 3 — Fine-tune EfficientNetB3 for individual tiger identification
py -3 src/tiger_ai_ultimate.py --train --stage identification --train-data data/processed --num-tigers 50
```

> **Note:** Stages 2 and 3 require TensorFlow (`pip install tensorflow`).
> Inference (below) runs on PyTorch only — TensorFlow is not needed for inference.

### Inference (run on any image folder)

```bash
# Batch run on a folder of images
py -3 src/tiger_ai_ultimate.py --data "path/to/your/images"

# Batch run with Grad-CAM explainability heatmaps
py -3 src/tiger_ai_ultimate.py --data "path/to/your/images" --gradcam

# Single image analysis
py -3 src/tiger_ai_ultimate.py --image tiger_photo.jpg

# Generate interactive HTML dashboard after a batch run
py -3 reports/generate_html_dashboard.py

# Generate Word run report
py -3 docs/generate_run_report_word.py
```

### First run notes
- `yolov8n-oiv7.pt` and `yolov8n.pt` are downloaded automatically on first run (~6 MB each).
- ResNet50 ImageNet weights are downloaded automatically via PyTorch hub.
- Internet connection required for first run only. All subsequent runs use local cache.

---

## Repository Structure

```
AI-Projects / EPAIB-Group-4 branch
|
+-- src/
|   +-- tiger_ai_ultimate.py      ← SINGLE END-TO-END FILE (~3000 lines)
|   |                                Section 0 : Training
|   |                                             - YOLOv8 custom fine-tune (--stage yolo)
|   |                                             - ResNet50 detection (--stage detection)
|   |                                             - EfficientNetB3 individual ID (--stage identification)
|   |                                Section 1 : Setup & Configuration
|   |                                Section 2 : Image Triage
|   |                                Section 3 : Preprocessing (CLAHE + Dehazing + IR normalisation)
|   |                                Section 4 : Model Loading (YOLOv8 OIV7 + COCO + ResNet50)
|   |                                Section 5 : Tiger Identity Database (JSON persistence)
|   |                                Section 6 : Species Verification Layer (ResNet50 top-3 gate)
|   |                                Section 7 : ViewPoint Classification (Left/Right/Frontal)
|   |                                Section 8 : Colour Morph Classification
|   |                                Section 8b: Water Reflection Guard
|   |                                Section 9 : Grad-CAM Explainability
|   |                                Section 10: Artefact Guards (shadow, corner trace, overlap)
|   |                                Section 11: Cascade Detection (OIV7 → COCO → whole-image)
|   |                                Section 12: Single Image Analysis
|   |                                Section 13: Batch Processing + ViewPoint-Aware Population Report
|   |
|   +-- devils_advocate.py        ← Validation tool — runs pipeline twice on same images
|   |                                and compares results to verify consistency
|   |
|   +-- yeshvir/                  ← Team member contributions (PyTorch scripts)
|       +-- yeshvir_colab_pytorch.py
|       +-- yeshvir_multi_animal_pytorch.py
|
+-- data/
|   +-- sample/tigers/
|       +-- train/                 Training images (committed for testing)
|       +-- test/                  Test images
|       +-- val/                   Validation images
|
+-- reports/
|   +-- generate_html_dashboard.py    Interactive HTML dashboard generator
|   +-- Tiger_AI_Dashboard.html       [OUTPUT — open in browser]
|   +-- Tiger_AI_Ultimate_Run_Report.docx  [OUTPUT — Word report]
|   +-- ultimate_population_report.csv     [OUTPUT — per-detection log]
|   +-- ultimate_tiger_db.json             [OUTPUT — tiger identity DB]
|   +-- ultimate_annotated/               [OUTPUT — annotated images]
|   +-- ultimate_gradcam/                 [OUTPUT — Grad-CAM heatmaps]
|
+-- docs/
|   +-- architecture.md               Pipeline design + design decisions
|   +-- amur_tiger_run_observations.md Detailed run observations (ATRW dataset)
|   +-- algorithm_stepbystep.md        Step-by-step explanation (non-technical)
|   +-- capstone_checklist.md          Submission checklist for IIM faculty
|
+-- presentation/
|   +-- TRACE_Presentation.pptx        Final presentation deck
|   +-- Tiger_Enumeration_Group4.pptx  Group 4 submission deck
|
+-- notebooks/                         Exploration notebooks (Colab)
+-- requirements.txt                   All Python dependencies
+-- README.md                          This file
```

---

## Design Rules (Non-Negotiable)

1. **Scale to 1 lakh images** — architecture must handle national deployment. Current version uses in-memory DB; FAISS vector index is the next step.
2. **Training and test data always change** — `--data` flag means zero code changes to run on any new dataset.
3. **Tested on different datasets** — validated on Sundarbans (187 images) and ATRW Amur Tiger benchmark (3,392 images) with identical code.
4. **Grad-CAM is mandatory** — every detection must produce visual proof the model reads stripe patterns. "A model you cannot explain is a model you cannot trust." — Prof. Mahesh Balan, IIM Lucknow.

---

## Datasets Used

| Dataset | Source | Size | Ground Truth |
|---|---|---|---|
| Sundarbans Training Set | Group 4 collection | 187 images | Manually verified |
| ATRW — Amur Tiger Re-ID | ICCV Wildlife CV Challenge | 3,392 train + test | Yes — reid_list_train.csv (92 tigers) |
| ImageNet | Stanford Vision Lab (via PyTorch) | Used via pretrained ResNet50 | Yes (1,000 classes) |

**ATRW Citation:**
> Li, S., Li, J., Tang, H., Qian, R., & Wang, W. (2019). ATRW: A Benchmark for Amur Tiger Re-identification in the Wild. *ACM Multimedia 2020.* arXiv:1906.05586

---

## Programme Details

| | |
|---|---|
| Programme | Executive Programme in AI for Business (EPAIB) |
| Institution | IIM Lucknow |
| Batch | 05 |
| Group | 4 |

## Team — Group 4

| # | Name |
|---|---|
| 1 | Mayank Tandon |
| 2 | Kuldeep Jain |
| 3 | Bhargavi Meduri |
| 4 | Anamitra Lahiri |
| 5 | Yeshvir Singh |
| 6 | Ashish |
| 7 | Sumani Goyal |
| 8 | Amit |
| 9 | Muthu |
| 10 | Srinivasa Murthy |

---

*Built for the tigers of Sundarbans — and every reserve in India.*
