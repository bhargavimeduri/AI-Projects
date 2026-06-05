# Meeting Notes — Group 4
## EPAIB Batch 05 | IIM Lucknow

---

## Meeting Log

---

### Meeting 1
**Date:** 2026-05-19
**Attendees:** [List names]
**Agenda:** Project kickoff

**Decisions Made:**
- Project topic confirmed: Sundarban Tiger Image Processing
- Repo created: epaib-batch05-group4
- Tech stack agreed: Python, TensorFlow, YOLOv8

**Action Items:**
| Task | Owner | Due |
|------|-------|-----|
| Source dataset from Kaggle/iNaturalist | [Name] | [Date] |
| Set up virtual environment & test requirements.txt | All | [Date] |
| Review architecture.md and suggest changes | All | [Date] |
| Fill in team member details in README.md | Bhargavi | [Date] |

**Next Meeting:** [Date & Time]

---

### Meeting 2 — Pipeline Build Sprint
**Date:** 2026-05-30
**Attendees:** Group 4 (Bhargavi Meduri + teammates)
**Agenda:** Review what has been built, prepare for Sowmya mam review

---

#### What We Built

**Dataset**
- 282 tiger images collected across 3 splits:
  - `train/` — 187 images (pipeline runs here)
  - `val/`   — 27 images (available for FNR evaluation)
  - `test/`  — 68 images (blind final evaluation)
- Formats: PNG, JPEG, WEBP supported
- Annotations: `train_annotations.csv`, `val_annotations.csv`, `test_annotations.csv`

---

**Core Pipeline — `src/tiger_pipeline.py` (v4)**

Full 3-model pipeline built end-to-end in a single Python file:

| Phase | What it does |
|-------|-------------|
| Phase 1 — Triage | Rejects blurry, overexposed, underexposed images. Detects IR/night images. |
| Phase 2a — Preprocessing | CLAHE contrast enhancement + Dark Channel Prior dehazing (fog/rain removal) |
| Phase 2b — Detection + Identity | YOLOv8n detects animals → ResNet50 extracts 2048-dim embedding → cosine similarity assigns Tiger_ID |
| Phase 3 — Classification | HSV analysis classifies color morph. Laplacian variance estimates subspecies (placeholder). |

Key design decisions made:
- **YOLOv8n** (not fine-tuned) — uses COCO proxy classes {15,16,17,24} since COCO has no tiger class
- **ResNet50** — final FC layer removed, keeps 2048-dim feature vector as tiger fingerprint
- **Cosine similarity** (not Euclidean) — scale-invariant, handles day/night lighting differences
- **Similarity threshold: 0.83** — tigers above this score = same individual
- **Partial body handling** — tail/leg crops used for matching only, not to update identity DB
- **Running average embedding** — DB embedding improves with each full-body sighting
- **CenterCrop(224)** not Resize(224,224) — preserves aspect ratio, ImageNet standard

CSV output columns: `Image_File, Image_Quality, Tigers_In_Frame, Tiger_ID, Is_New_Tiger, Is_Partial_View, Color_Morph, Subspecies, Match_Score, Detection_Conf, Skipped_Reason`

**Results on train set (187 images):**
- 13 unique tigers identified
- Color morphs detected: Orange Standard (majority), White, Black
- Devil's Advocacy: PASSED (see below)

---

**Validation — `src/devils_advocate.py`**

Ran the full pipeline TWICE from scratch (fresh empty DB each time) on the same 187 images.

| Check | Result |
|-------|--------|
| Unique tiger count | Run 1: 13, Run 2: 13 — CONSISTENT |
| Per-image tiger count | 32/32 images matched — CONSISTENT |
| Color morph per image | 35/35 detections matched — CONSISTENT |
| Mean match score | 0.8174 both runs — CONSISTENT |
| **VERDICT** | **PASSED — fully reproducible** |

---

**Explainability — `src/gradcam_check.py`**

Grad-CAM implemented on ResNet50 layer4 to visually verify where the model looks.

- Finds first 10 tiger crops via YOLO
- Generates side-by-side: Original | Heatmap | Overlay
- ResNet top classes returned: 281/282/285/292 (ImageNet cat/tiger family) — model is recognising correct animal family
- Output saved to: `reports/gradcam_outputs/`
- Does NOT modify `tiger_pipeline.py`

Principle recorded: *"A model you cannot explain is a model you cannot trust."* — Prof. Mahesh Balan Sir

---

**Presentation — `presentation/build_ppt.py`**

14-slide PowerPoint built programmatically using python-pptx:

| Slide | Content |
|-------|---------|
| 1 | Title — TRACE system |
| 2 | Problem Statement (expanded) |
| 3 | Camera Trap Challenges (4 categories) |
| 4 | 3-Model Solution Approach |
| 5–14 | Architecture, results, pipeline flow, evaluation, next steps |

Colour palette: Blue (not green) — `#0D2B55` dark, `#1A4F8A` mid, `#E3F2FD` light

---

**Google Colab Notebook — `TRACE_Tiger_Pipeline.ipynb`**

12-cell notebook for team sharing — runs the full pipeline in Google Colab with no local setup:
- Installs all dependencies
- Mounts Google Drive
- Runs triage → preprocessing → detection → identity → classification
- Saves CSV report
- Downloads results

---

**Professor Code Reviews — `presentation/professor_reviews/`**

Pipeline reviewed by all 5 EPAIB faculty (simulated):

| Professor | Key Feedback |
|-----------|-------------|
| Prof. Mahesh Balan (Deep Learning) | Add Grad-CAM ✓, measure FNR, calibrate threshold from ROC curve |
| Prof. Sowmya (Faculty Supervisor) | No ground truth labels, subspecies is placeholder, explain 13 vs 51 tiger count |
| Prof. Laxmi Narayan (AI Strategy) | Add Gradio UI, benchmark processing speed, add metadata (timestamp, camera ID) |
| Prof. Prithviraj (Data Strategy) | Domain shift risk, analyse filter rejection rate |
| Prof. Sumit Singh (Product/GenAI) | Add bounding box visualisation, simplify Colab Cell 3 |

---

#### Open Issues / Next Steps

| Issue | Priority | Notes |
|-------|----------|-------|
| Similarity threshold 0.83 not calibrated | High | Need ROC curve on labelled val images |
| No ground truth — cannot compute true FNR | High | Manually annotate ~30 val images |
| Subspecies classifier is placeholder | Medium | Needs dedicated ML model |
| Domain shift: Google Images vs real camera traps | Medium | Retrain on camera trap data when available |
| DB is in-memory only | Medium | Add save/load to JSON or SQLite |
| No Gradio UI | Low | Add drag-and-drop upload interface |
| Bounding box visualisation missing from Colab | Low | Add cell to draw boxes on output images |
| architecture.md was outdated | Done ✓ | Updated 2026-05-30 |
| problem_statement.md missing unique count + morph objectives | Done ✓ | Updated 2026-05-30 |

---

**Next Meeting:** [Date & Time]

---

> Add a new section for every meeting. Keep it brief — decisions + action items only.

---

*Last updated: 2026-05-30 | Group 4 | EPAIB Batch 05 | IIM Lucknow*
