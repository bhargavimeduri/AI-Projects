# System Architecture
## TIGER AI ULTIMATE — v3.0 Fused Multi-Source Pipeline
## EPAIB Batch 05 | Group 4 | IIM Lucknow

---

## DESIGN RULES (Non-Negotiable)

These rules apply to every version of this pipeline, now and in future.

| Rule | What It Means | Why |
|---|---|---|
| **Must scale to 1,00,000 images** | The pipeline cannot assume a small dataset. Every design decision must hold when run on 1 lakh camera trap images from 50+ tiger reserves across India. | Real-world WII (Wildlife Institute of India) deployment requirement. Sundarbans alone has thousands of camera-days per survey. |
| **Training and test data will always change** | Never hardcode paths, class lists, or threshold values as if the dataset is fixed. Every parameter must be configurable. | Different survey seasons, different cameras, different forests = different images every time. A pipeline tuned only on our 187 training images is not a real system. |
| **Model will be tested on different datasets for finetuning** | The pipeline must be re-runnable on new datasets without rewriting code. Output format must remain consistent regardless of input dataset. | Finetuning on new labelled data (when WII provides it) is part of the roadmap. The architecture must accept this without structural changes. |
| **Model you cannot explain is a model you cannot trust** | Grad-CAM is not optional. Every tiger detection must be explainable — the model must be looking at stripes, not background. | Prof. Mahesh Balan's core principle. Also a regulatory requirement for AI systems used in conservation policy decisions. |
| **Never count sightings — count individuals** | 10 images of the same tiger = 1 tiger in the census. Identity matching is the core output, not detection count. | The entire purpose of the project is census (how many unique tigers), not detection (how many tiger images). |
| **Every run must be reproducible** | Same images in = same tiger IDs out. Deterministic behaviour. | Reproducibility is a scientific requirement for conservation surveys. |

---

## Pipeline Version History

| Version | File | Key Addition |
|---|---|---|
| v1.0 (TRACE) | `tiger_pipeline.py` | Base pipeline: YOLO + ResNet50 + cosine similarity |
| v2.0 (TRACE Complete) | `tiger_ai_complete.py` | + Shadow detection, corner trace, overlap guard, Grad-CAM, JSON DB persistence, running average |
| v3.0 (Ultimate) | `tiger_ai_ultimate.py` | + Cascade detection (OIV7), species verification (top-3 ResNet), viewpoint classification (L/R/F), viewpoint-aware census, IR grayscale normalisation |

---

## v3.0 Pipeline Architecture (tiger_ai_ultimate.py)

Sources fused into v3.0:
- **TRACE (v2.0)** — base architecture, all guards, running average DB
- **Dr. Bose** — OIV7 detector, grayscale IR normalisation, viewpoint-aware census
- **Yeshvir Script B** — species verification (top-3 ResNet), whole-image fallback
- **Yeshvir Script A** — best-match algorithm (no break bug)

```
CAMERA TRAP IMAGE
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│   PHASE 1 — IMAGE TRIAGE                                │
│   - Blur: Laplacian variance < 80   → Unusable_Blurry  │
│   - Overexpose: mean grey > 240     → Unusable_Over    │
│   - Underexpose: mean grey < 20     → Unusable_Under   │
│   - IR/Night: R≈G≈B channels       → IR_Night          │
│   Output: PASS / SKIP               (cheapest gate)    │
└────────┬────────────────────────────────────────────────┘
         │ PASS
         ▼
┌─────────────────────────────────────────────────────────┐
│   PHASE 2 — PREPROCESSING CHAIN                         │
│                                                         │
│   If IR_Night:                                          │
│   └─ Grayscale normalisation (Dr. Bose)                 │
│      Convert → grey → 3-channel BGR                     │
│      Forces model to read stripe geometry, not colour   │
│                                                         │
│   CLAHE (contrast enhancement — from TRACE)             │
│   └─ Applied to L-channel of LAB colourspace            │
│      clipLimit=2.0, tileGridSize=(8×8)                  │
│                                                         │
│   Dark Channel Prior Dehazing (fog removal — from TRACE)│
│   └─ Removes monsoon haze, fog, rain scatter            │
│      Critical for Sundarbans humid conditions           │
└────────┬────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│   PHASE 3 — CASCADE DETECTION (3-Stage Waterfall)       │
│                                                         │
│   STAGE 1: YOLOv8n-OIV7 (Dr. Bose)                    │
│   └─ Open Images V7 — has actual "Tiger" class          │
│      Confidence: 0.30                                   │
│      If Tiger class detected → use these boxes          │
│      Requires verify: NO (OIV7 already confirmed)       │
│             │                                           │
│             │ If OIV7 finds nothing                     │
│             ▼                                           │
│   STAGE 2: YOLOv8n-COCO (TRACE)                        │
│   └─ Proxy classes: 15=cat, 16=dog, 17=horse, 24=zebra │
│      Confidence: 0.35                                   │
│      Requires verify: YES (proxy class ≠ tiger)         │
│             │                                           │
│             │ If COCO also finds nothing                │
│             ▼                                           │
│   STAGE 3: Whole-Image Fallback (Yeshvir Script B)     │
│   └─ Use full image as one detection box                │
│      is_fallback = True                                 │
│      Requires verify: YES (only species gate we have)   │
└────────┬────────────────────────────────────────────────┘
         │ One or more candidate boxes
         ▼
┌─────────────────────────────────────────────────────────┐
│   PHASE 4 — PER-CROP GUARDS (all from TRACE v2.0)       │
│                                                         │
│   Size filter: box area < 3% of image → skip            │
│   Shadow guard: HSV saturation < 30 AND brightness < 85 │
│     → achromatic dark blob = tiger shadow not tiger      │
│   Corner trace: box touching image border within 12px    │
│     → label Corner_Trace, match-only (no DB update)     │
│   Overlap guard: new box IoU > 45% with accepted box    │
│     → duplicate / shadow artefact, reject               │
└────────┬────────────────────────────────────────────────┘
         │ Clean crop
         ▼
┌─────────────────────────────────────────────────────────┐
│   PHASE 5 — SPECIES VERIFICATION (Yeshvir Script B)     │
│                                                         │
│   Applies to COCO proxy boxes AND whole-image fallback. │
│   OIV7 boxes skip this phase (already confirmed).       │
│                                                         │
│   ResNet50 top-3 classification                         │
│   └─ If "tiger" in any of top-3 labels → ACCEPT        │
│      If not → REJECT crop entirely                      │
│   This is the second model's vote — both YOLO and       │
│   ResNet must agree before we spend time fingerprinting.│
└────────┬────────────────────────────────────────────────┘
         │ Verified tiger crop
         ▼
┌─────────────────────────────────────────────────────────┐
│   PHASE 6 — VIEWPOINT CLASSIFICATION (Dr. Bose)         │
│                                                         │
│   Every accepted crop gets a viewpoint label:           │
│   Left_Flank  — crop centre in left 40% of image        │
│   Right_Flank — crop centre in right 40% of image       │
│   Frontal     — centre or tall aspect ratio (h > 1.4w)  │
│                                                         │
│   WHY: A tiger's left stripe pattern ≠ right stripe     │
│   pattern. Without this, Tiger_A seen from both sides   │
│   could be assigned two different Tiger_IDs.            │
└────────┬────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│   PHASE 7 — IDENTITY MATCHING (TRACE + Yeshvir fix)     │
│                                                         │
│   ResNet50 feature extractor (2048-dim avg_pool)        │
│   Compare against all known tigers in DB                │
│   BEST match algorithm — full loop, no early break      │
│   (break bug fix from Yeshvir Script A)                 │
│                                                         │
│   Score >= 0.83 → existing tiger                        │
│   └─ If full-body: update running average embedding     │
│      If partial/corner: match-only, never modify DB     │
│   Score <  0.83 → new tiger (enroll in DB)              │
│                                                         │
│   Running average update:                               │
│   new = (old × n + fingerprint) / (n + 1)              │
│   DB improves with every sighting.                      │
└────────┬────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│   PHASE 8 — COLOUR MORPH CLASSIFICATION (TRACE)         │
│   HSV analysis: Orange / White / Golden / Black /       │
│                 Snow White                              │
└────────┬────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│   PHASE 9 — GRAD-CAM EXPLAINABILITY (TRACE)             │
│   ResNet50 layer4 hooks                                 │
│   Output: Original | Heatmap | Overlay                  │
│   Pass = red heatmap on stripes                         │
│   Fail = red heatmap on background (fingerprint bad)    │
└────────┬────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│   PHASE 10 — OUTPUT                                     │
│   CSV: per-detection (Tiger_ID, ViewPoint, Source, ...) │
│   JSON DB: running average embeddings + sighting counts │
│   Annotated images with bounding boxes                  │
│   Grad-CAM outputs                                      │
│   Text population report with viewpoint-aware census    │
└─────────────────────────────────────────────────────────┘
```

---

## Viewpoint-Aware Census (Dr. Bose's Method)

A tiger's left and right flank have different stripe patterns.
If we see Tiger_A from the left in 3 images and from the right in 2 images,
a naive counter may assign two IDs. The viewpoint-aware method corrects for this.

```
After processing all images:

For each Tiger_ID in DB:
  Collect all viewpoints seen: {Left_Flank, Right_Flank, Frontal}

left_tigers   = unique IDs seen from Left_Flank
right_tigers  = unique IDs seen from Right_Flank
frontal_tigers = unique IDs seen Frontally

CONSERVATIVE estimate = max(left, right, frontal)
  → avoids counting same tiger's left+right as 2 tigers
  → safe lower bound for population report

LIBERAL estimate = union of all IDs
  → upper bound (assumes every ID is truly unique)

Best practice: report both and explain the difference to the panel.
```

---

## Model Architecture Details

### YOLOv8n-OIV7 (Stage 1 Detector)
```
Model:     yolov8n-oiv7.pt (Ultralytics, Open Images V7)
Classes:   601 classes including direct "Tiger" class
Confidence: 0.30 (slightly lower — model is more trustworthy on Tiger class)
Role:      Primary detector — if it fires, no proxy needed
Source:    Dr. Bose's architecture recommendation
```

### YOLOv8n-COCO (Stage 2 Detector — Fallback)
```
Model:     yolov8n.pt (Ultralytics, COCO 80 classes)
Tiger proxy: 15=cat, 16=dog, 17=horse, 24=zebra
             (COCO has no Tiger class)
Confidence: 0.35 (higher — proxy matches have more noise)
Role:      Secondary detector — catches tigers OIV7 misses
           (e.g. occluded, distant, unusual angle)
Source:    TRACE v1.0 (Group 4)
```

### ResNet50 — Species Gate (top-3 check)
```
Model:     ResNet50 (torchvision, ImageNet pretrained)
Input:     224×224 tiger crop
Output:    Top-3 predicted labels from 1000 ImageNet classes
Check:     Does any of top-3 labels contain the word "tiger"?
Apply to:  COCO proxy detections and whole-image fallback ONLY
Source:    Yeshvir Script B
```

### ResNet50 — Identity Extractor (2048-dim fingerprint)
```
Model:     ResNet50 (same weights, FC layer removed)
Output:    2048-dimensional embedding from avg_pool layer
Matching:  Cosine similarity (scale-invariant — day/night consistent)
Threshold: 0.83 (calibrated on our dataset — may need re-tuning on new data)
DB update: Running average per tiger — improves with each sighting
Source:    TRACE v1.0 + Yeshvir (same concept)
```

---

## Scale Requirements (1,00,000 Images)

The v3.0 pipeline handles small datasets today (187 images).
For production at 1 lakh image scale, these upgrades are needed.

| Component | Current (v3.0) | Required for 1L Images |
|---|---|---|
| YOLO detection | Serial, per-image | Batch inference (batch_size=32) on GPU |
| ResNet features | Serial, per-crop | Batch feature extraction |
| Vector matching | In-memory list | FAISS IVF index (Dr. Bose recommendation) |
| DB storage | JSON file | SQLite or PostgreSQL |
| Empty image filter | YOLO Stage 1 | MegaDetector (Microsoft AI for Earth) as Stage 0 |
| Processing time (CPU) | ~187 images / 10 min | 1L images = 90+ hours (needs GPU) |
| Processing time (GPU) | ~187 images / 2 min | 1L images = ~2 hours (RTX 3060+) |

**MegaDetector as Stage 0 (for 1L scale):**
Run MegaDetector first to discard "empty" images (wind, leaves, shadows).
~90% of camera trap images are empty. Running MegaDetector first reduces
the 1L images to ~10,000 actual animal images before any tiger-specific work.

---

## Finetuning on New Datasets

When WII provides labelled tiger data (tiger ID annotations on images):

### Phase 2 Roadmap — EfficientNetB3 Fine-Tuning
```python
# Phase 1: Freeze base, train new head on labelled tiger images
# - 10 epochs, lr=1e-3

# Phase 2: Unfreeze top 30 layers, fine-tune
# - 20 epochs, lr=1e-5

# This is the blueprint from the Colab script (copy_of_tiger_ai_system_complete.py)
# It replaces the pretrained ResNet50 with a tiger-specific backbone.
```

When new datasets arrive:
1. Run the pipeline on the new dataset with `--data path/to/new/data`
2. The DB starts fresh (or can be loaded from a previous run)
3. Evaluate: run `devils_advocate.py` for reproducibility check
4. Calibrate: plot cosine similarity score distribution → adjust SIMILARITY_THRESHOLD

---

## Technology Stack

| Component | Technology | Why |
|---|---|---|
| Stage 1 detection | YOLOv8n-OIV7 (Ultralytics) | Direct Tiger class, no proxy needed |
| Stage 2 detection | YOLOv8n-COCO (Ultralytics) | Fallback, proxy classes |
| Species verification | ResNet50 top-3 (PyTorch) | Second-vote species gate |
| Feature extraction | ResNet50 2048-dim (PyTorch) | Proven embeddings, ImageNet pretrained |
| Identity matching | Cosine similarity (sklearn) | Scale-invariant across lighting conditions |
| DB (current) | JSON file | Simple, persistent, cross-session |
| DB (1L scale) | FAISS + SQLite | Fast vector search, structured storage |
| Preprocessing | OpenCV CLAHE + Dark Channel Prior | Camera trap specific quality issues |
| IR normalisation | Greyscale→3ch (Dr. Bose) | Night/day feature consistency |
| Color morph | HSV analysis (OpenCV) | Rule-based, fast, no training needed |
| Explainability | Grad-CAM (PyTorch hooks) | Visual proof of model attention |
| Output | CSV + JSON + annotated images | Shareable, reproducible |

---

## Data Flow

```
Camera trap images (any quantity, any season)
    │
    ├── train/   ──────────────────────► Pipeline runs here (builds DB)
    ├── val/     ──────────────────────► FNR evaluation (annotate 30 images)
    └── test/    ──────────────────────► Blind final evaluation
```

**Important:** The pipeline is dataset-agnostic.
It reads from `--data` argument. No code change needed for new datasets.

---

## Known Limitations and Next Steps

| Limitation | Impact | Fix |
|---|---|---|
| OIV7 model may not detect occluded tigers | Missed detections in dense vegetation | Fine-tune on wildlife camera trap data |
| Similarity threshold 0.83 not ROC-calibrated | May over/under-split identities | Plot ROC curve on annotated val set |
| Viewpoint classifier is heuristic (position-based) | May misclassify oblique shots | Train a dedicated pose classifier |
| No MegaDetector (empty-image filter) | At 1L scale, wastes 90% compute on empty images | Add as Stage 0 when deploying at scale |
| Subspecies classifier is a placeholder | Subspecies output unreliable | Train dedicated CNN on Bengal/Siberian/Indochinese labelled data |
| Domain shift (ImageNet vs camera trap) | Fingerprints may not generalise | Fine-tune ResNet on camera trap tiger images (EfficientNetB3 Phase 2) |
| FNR not measured | Don't know how many tigers we miss | Manually annotate 30 val images with confirmed Tiger_IDs |
| FAISS not integrated | In-memory list slow at scale | Integrate faiss-cpu when dataset > 5,000 images |

---

## File Map

```
src/
  tiger_ai_ultimate.py    ← v3.0 FUSED pipeline (this document)
  tiger_ai_complete.py    ← v2.0 TRACE pipeline (shadow/corner/overlap guards)
  tiger_pipeline.py       ← v1.0 base pipeline
  yeshvir/
    yeshvir_colab_pytorch.py           ← Yeshvir Script A (PyTorch port)
    yeshvir_multi_animal_pytorch.py    ← Yeshvir Script B (PyTorch port)

docs/
  architecture.md         ← THIS FILE — always update when pipeline changes
  yeshvir_code_review.md  ← Review of Yeshvir's two scripts
  project_work_summary.md ← Plain English project history

reports/
  tiger_population_report.csv    ← TRACE v2.0 output
  ultimate_population_report.csv ← v3.0 fused output
  tiger_registry_db.json         ← TRACE v2.0 identity DB
  ultimate_tiger_db.json         ← v3.0 identity DB
  panel_feedback_report.xlsx     ← Professor feedback + Yeshvir review
```

---

*Last updated: 2026-06-02 | Tiger AI Ultimate v3.0 | Group 4 | EPAIB Batch 05 | IIM Lucknow*
*RULE: Update this file every time the pipeline changes. Architecture.md is the single source of truth.*
