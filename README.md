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

### Run on your dataset

```bash
# Run on a folder of images (batch mode)
py -3 src/tiger_ai_ultimate.py --data "path/to/your/images"

# Run with Grad-CAM explainability images
py -3 src/tiger_ai_ultimate.py --data "path/to/your/images" --gradcam

# Run on a single image
py -3 src/tiger_ai_ultimate.py --image tiger_photo.jpg

# Generate interactive HTML dashboard (after a batch run)
py -3 reports/generate_html_dashboard.py

# Generate Word run report
py -3 docs/generate_run_report_word.py
```

### First run notes
- On first run, `yolov8n-oiv7.pt` (~6MB) will be downloaded automatically from Ultralytics.
- `yolov8n.pt` and ResNet50 weights are also downloaded automatically on first use.
- Internet connection required for first run. All subsequent runs use local cache.

---

## Repository Structure

```
epaib-batch05-group4/
|
+-- src/
|   +-- tiger_ai_ultimate.py      [MAIN PIPELINE — v3.0 fused]
|
+-- reports/
|   +-- generate_html_dashboard.py   Interactive HTML dashboard generator
|   +-- Tiger_AI_Dashboard.html      [OUTPUT — open in browser]
|   +-- Tiger_AI_Ultimate_Run_Report.docx  [OUTPUT — Word report]
|   +-- ultimate_population_report.csv     [OUTPUT — per-detection log]
|   +-- ultimate_tiger_db.json             [OUTPUT — tiger identity database]
|   +-- Panel_Code_Review_Feedback.xlsx    [OUTPUT — panel review]
|   +-- ultimate_annotated/               [OUTPUT — annotated images]
|   +-- ultimate_gradcam/                 [OUTPUT — Grad-CAM heatmaps]
|
+-- docs/
|   +-- architecture.md               Pipeline design + design rules
|   +-- amur_tiger_run_observations.md Detailed run observations (ATRW)
|   +-- capstone_checklist.md          Submission checklist for IIM faculty
|   +-- dataset_description.md         Dataset sources and citations
|   +-- generate_run_report_word.py    Word report generator
|   +-- generate_panel_feedback_excel.py  Excel feedback sheet generator
|   +-- build_ppt.py                   PowerPoint deck builder
|
+-- notebooks/                        Google Colab notebooks
+-- data/
|   +-- sample/tigers/train/          Sample images (committed for testing)
|
+-- requirements.txt                  All Python dependencies
+-- README.md                         This file
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
| Project Lead | Bhargavi Meduri |

---

*Built for the tigers of Sundarbans — and every reserve in India.*
