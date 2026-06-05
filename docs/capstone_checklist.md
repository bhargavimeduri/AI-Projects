# EPAIB Batch 05 — Group 4 | Capstone Project Submission Checklist
## Tiger Enumeration AI System — IIM Lucknow

**Project:** TRACE — Tiger Recognition and Census Engine  
**Programme:** Executive Programme in AI for Business (EPAIB), Batch 05, IIM Lucknow  
**Group 4 | Date:** 2026-06-02

---

## HOW TO USE THIS CHECKLIST

For each item:
- **[ ]** = Not started
- **[WIP]** = Work in progress
- **[DONE]** = Completed and file exists

Before submission, every item in **MUST HAVE** sections must be **[DONE]**.

---

## SECTION A — CORE DELIVERABLES (MUST HAVE)

### A1. Source Code

| # | Item | Status | File / Location |
|---|---|---|---|
| A1.1 | Main pipeline code — well commented | [WIP] | `src/tiger_ai_ultimate.py` |
| A1.2 | HTML report generator | [DONE] | `reports/generate_html_dashboard.py` |
| A1.3 | Word report generator | [DONE] | `docs/generate_run_report_word.py` |
| A1.4 | Panel feedback Excel generator | [DONE] | `docs/generate_panel_feedback_excel.py` |
| A1.5 | PPT builder script | [DONE] | `docs/build_ppt.py` |
| A1.6 | All code in a single zip or GitHub repo | [ ] | — |

**Code quality checklist (per file):**
- [ ] Every function has a docstring explaining what it does, what it takes, what it returns
- [ ] Every non-obvious line has an inline comment explaining WHY (not just what)
- [ ] No hardcoded Windows paths — use `--data` flag or config at top of file
- [ ] No dead code (commented-out old blocks)
- [ ] Import block at top is clean and ordered

---

### A2. Presentation Deck (PPT)

| # | Item | Status | File / Location |
|---|---|---|---|
| A2.1 | Cover slide — project name, team, programme | [DONE] | Auto-built by `build_ppt.py` |
| A2.2 | Problem statement — why tiger enumeration matters | [DONE] | Slide 2 |
| A2.3 | Dataset description — what data we used, why | [WIP] | Slide 3 — needs ATRW details |
| A2.4 | Architecture diagram — full pipeline visual | [DONE] | Slide 4 |
| A2.5 | Algorithm details — OIV7, COCO, ResNet50, cosine | [WIP] | Needs "why this algorithm" for each |
| A2.6 | Why each model was chosen (justify the choice) | [ ] | New slide needed |
| A2.7 | Results — Sundarbans run | [DONE] | Slide 6 |
| A2.8 | Results — Amur Tiger (ATRW) run | [ ] | New slide needed |
| A2.9 | Accuracy / evaluation metrics | [ ] | New slide — FNR, precision, recall |
| A2.10 | Bugs found and fixed | [DONE] | Slide in deck |
| A2.11 | Lessons learned + next steps | [DONE] | Last slide |
| A2.12 | References / citations | [ ] | New slide needed |

**PPT quality checklist:**
- [ ] Consistent font and colour theme throughout
- [ ] No slide has more than 6 bullet points (visuals > text)
- [ ] Algorithm slides show a diagram or flowchart, not just text
- [ ] Every result slide states: what metric, what the number means, what decisions it drives
- [ ] Grad-CAM images included as visual proof the model reads stripes
- [ ] Slide count: aim for 12–15 slides (not more)

---

### A3. Reports and Outputs

| # | Item | Status | File / Location |
|---|---|---|---|
| A3.1 | HTML Interactive Dashboard | [DONE] | `reports/Tiger_AI_Dashboard.html` |
| A3.2 | Word Run Report | [DONE] | `reports/Tiger_AI_Ultimate_Run_Report.docx` |
| A3.3 | Population Census CSV | [DONE] | `reports/ultimate_population_report.csv` |
| A3.4 | Tiger Identity Database (JSON) | [DONE] | `reports/ultimate_tiger_db.json` |
| A3.5 | Panel Feedback Excel | [DONE] | `reports/Panel_Code_Review_Feedback.xlsx` |
| A3.6 | Classified output images | [DONE] | `Amur Tigers/Classified_TestBh/` |

---

### A4. Documentation

| # | Item | Status | File / Location |
|---|---|---|---|
| A4.1 | Architecture document | [DONE] | `docs/architecture.md` |
| A4.2 | Amur Tiger run observations | [DONE] | `docs/amur_tiger_run_observations.md` |
| A4.3 | Dataset documentation | [ ] | `docs/dataset_description.md` (needed) |
| A4.4 | README — how to run the pipeline | [ ] | `README.md` (needed) |
| A4.5 | Requirements file | [ ] | `requirements.txt` (needed) |
| A4.6 | This checklist | [DONE] | `docs/capstone_checklist.md` |

---

## SECTION B — ALGORITHM JUSTIFICATION (MUST HAVE FOR IIM SUBMISSION)

This is the section professors most often ask about in vivas. Be ready to justify every model choice.

| Model | What it does | Why we chose it | Alternatives considered |
|---|---|---|---|
| **YOLOv8 + OIV7 weights** | Detects tiger bounding boxes in image | OIV7 has a direct "Tiger" class — no proxy needed. YOLOv8 is the current state-of-art real-time detector (2023) | MegaDetector (too complex to install), Faster R-CNN (slower, overkill) |
| **YOLOv8 + COCO weights** | Backup detection using proxy classes (cat, dog, zebra) | COCO has no Tiger class; cat/dog/zebra are closest shape proxies. Acts as Stage 2 fallback | Nothing — COCO is the standard benchmark |
| **ResNet50** | Species verification (top-3 gate) AND individual fingerprinting | Pretrained on ImageNet (includes tiger). Fast, well-understood. Proven transfer learning backbone | EfficientNetB3 (better accuracy, heavier), VGG16 (older, worse), MobileNet (too light for fine features) |
| **Cosine Similarity** | Matching tiger fingerprints across images | Scale-invariant (doesn't care about image brightness). Works in high-dimensional space. Standard in Re-ID literature | Euclidean distance (sensitive to scale), Siamese network (needs labelled pairs to train) |
| **Running Average Embedding** | Improves tiger fingerprint with each sighting | Single image can be noisy (partial body, angle). Average of all sightings is more robust | Fixed embedding from first sighting (less accurate), re-enroll only on high-confidence matches |
| **CLAHE** | Contrast enhancement before feature extraction | Tigers in shadow or fog have washed-out stripe contrast — CLAHE recovers texture without overexposing highlights | Histogram equalisation (global, crushes local detail), no preprocessing (worse features) |
| **Grad-CAM** | Visual explainability — shows where model looks | Regulatory/academic requirement. Proves the model reads stripes, not background. Prof-MB's non-negotiable | LIME (slower for images), SHAP (complex for CNNs), saliency maps (less interpretable) |

---

## SECTION C — DATASET DOCUMENTATION (MUST HAVE)

| Dataset | Source | Size | Purpose | Ground Truth? |
|---|---|---|---|---|
| **Sundarbans Training Set** | Collected by Group 4 | 187 images | Primary training/validation dataset | No (manually curated) |
| **ATRW — Amur Tiger Train** | ICCV Wildlife CV Challenge | 3,392 images | Large-scale validation and pipeline testing | Yes — `reid_list_train.csv` (92 unique tigers) |
| **Test Bh (Amur Tigers)** | Subset acquired by Group 4 | 35 images | Cross-dataset generalisation test | Partial |
| **ImageNet** | Stanford Vision Lab | 14M images (used via pretrained ResNet50) | Transfer learning base — includes tiger class | Yes (1,000 classes) |

**Dataset documentation must include:**
- [ ] How each dataset was acquired (ethics clearance if needed for wildlife data)
- [ ] Class distribution (how many images per tiger)
- [ ] Train/test split used
- [ ] Any augmentation applied
- [ ] Licence / citation for public datasets (ATRW paper citation needed)

**ATRW Citation (add to PPT references slide):**
> Li, S., Li, J., Tang, H., Qian, R., & Wang, W. (2019). ATRW: A Benchmark for Amur Tiger Re-identification in the Wild. *ACM Multimedia 2020.* arXiv:1906.05586

---

## SECTION D — EVALUATION METRICS (MUST HAVE)

| Metric | How to compute | Target | Current status |
|---|---|---|---|
| **True Positive Rate (TPR / Recall)** | TP / (TP + FN) | > 0.95 | Not measured against ground truth yet |
| **False Negative Rate (FNR)** | FN / (FN + TP) | < 0.05 | Not measured |
| **Precision** | TP / (TP + FP) | > 0.90 | Not measured |
| **F1 Score** | 2 × (P × R) / (P + R) | > 0.92 | Not measured |
| **Re-ID Accuracy** | % images matched to correct tiger (vs ATRW ground truth) | > 0.80 | Not measured — need to compare to `reid_list_train.csv` |
| **Blur rejection rate** | Blurry images / total images | Monitor | 25.9% (Amur), ~3% (Sundarbans) |
| **Mean match score** | Average cosine similarity for recaptures | > 0.85 | 0.796 (incl. new enrollments) |

**To compute accuracy against ATRW ground truth:**
```
py -3 src/evaluate_atrw_accuracy.py \
  --pred reports/ultimate_population_report.csv \
  --gt "C:\Users\91630\OneDrive\Desktop\Bhargavi AIB\Amur Tigers\train\reid_list_train.csv"
```
*(evaluation script to be written)*

---

## SECTION E — EXPLAINABILITY (MUST HAVE)

| Item | Status | Notes |
|---|---|---|
| Grad-CAM images showing model reading stripes | [ ] | Run `--gradcam` flag on 5–10 representative images |
| Grad-CAM on a NON-tiger image (negative case) | [ ] | Show what the model looks at when it says "not tiger" |
| Grad-CAM on a false positive (if any exist) | [ ] | Shows where model was fooled |
| Species gate rejection examples (top-3 labels) | [DONE] | In `amur_tiger_run_observations.md` Section 3 |
| ViewPoint classification examples | [ ] | Image + label for each of 3 classes |

**Grad-CAM run command:**
```
py -3 src/tiger_ai_ultimate.py --data "path/to/images" --gradcam
```
Grad-CAM images are saved to `reports/gradcam/`

---

## SECTION F — PRODUCTION READINESS (GOOD TO HAVE — SHOWS DEPTH)

| Item | Status | Notes |
|---|---|---|
| Scale analysis — what changes at 1L images | [DONE] | `docs/architecture.md` Scale Requirements section |
| FAISS integration plan | [DONE] | Documented in architecture.md |
| MegaDetector integration plan | [DONE] | Documented in architecture.md |
| GPU batch processing plan | [DONE] | Documented in architecture.md |
| Model card | [ ] | One-page summary of model, data, limitations, intended use |
| Ethical considerations | [ ] | Wildlife data privacy, false ID risk for conservation decisions |

---

## SECTION G — VIVA PREPARATION (MUST PREPARE)

These are the questions IIM faculty most commonly ask in capstone vivas. Prepare a 1–2 minute verbal answer for each.

### Architecture Questions
1. Why ResNet50 and not a more recent model like EfficientNet or ViT?
2. What is the difference between Stage 1, Stage 2, and Stage 3 of your cascade? Why do you need three stages?
3. Why cosine similarity? Why not Euclidean distance?
4. What is transfer learning? Why is it the right choice here given your dataset size?
5. What is Grad-CAM? Can you walk me through what it shows in your results?

### Results Questions
6. You found 13 tigers from 187 images. How confident are you that number is accurate? What is your margin of error?
7. What is your false negative rate? What does it mean for conservation if you miss a tiger?
8. Your mean match score is 0.796. Is that good? What does 0.83 threshold mean?
9. The Amur Tiger run showed 25.9% blurry images. What does that mean for real camera trap deployments?
10. You ran on two completely different datasets (Sundarbans and Amur) with zero code changes. What does that prove?

### Design Questions
11. You said the pipeline must scale to 1 lakh images. What specifically needs to change in the current code to support that?
12. The COCO proxy stage contributes only 2.1% of detections. Should it be removed? Justify your answer.
13. You found a bug mid-run (Corner Trace on fallback detections). What caused it? How did you fix it? What was the impact?
14. If WII (Wildlife Institute of India) gives you labelled data, what would you do differently?

### Business Impact Questions
15. What is the business value of this system vs. manual tiger surveys?
16. Who are the end users of this system and what decisions does it support?
17. What are the risks if the system makes a wrong identification?

---

## SECTION H — SUBMISSION PACKAGE (FINAL ZIP)

When everything above is done, the submission zip should contain:

```
Group4_TRACE_TigerEnumeration/
├── src/
│   └── tiger_ai_ultimate.py          (commented)
├── docs/
│   ├── architecture.md
│   ├── amur_tiger_run_observations.md
│   ├── dataset_description.md        (to create)
│   ├── capstone_checklist.md
│   └── README.md                     (to create)
├── reports/
│   ├── Tiger_AI_Dashboard.html       (interactive)
│   ├── Tiger_AI_Ultimate_Run_Report.docx
│   ├── ultimate_population_report.csv
│   └── Panel_Code_Review_Feedback.xlsx
├── presentation/
│   └── Group4_TRACE_Presentation.pptx
├── gradcam/
│   └── [5-10 Grad-CAM sample images]
└── requirements.txt                  (to create)
```

---

## PRIORITY ORDER — WHAT TO DO NEXT

| Priority | Task | Why |
|---|---|---|
| 1 | **Add line-by-line comments to `tiger_ai_ultimate.py`** | Code comments = 30% of technical grade. Non-tech panel needs English explanations |
| 2 | **Run Grad-CAM on sample images** | Prof-MB will ask for this. It's visual proof of explainability |
| 3 | **Measure accuracy against ATRW ground truth** | Actual accuracy number is the headline result |
| 4 | **Update PPT with algorithm justification slide** | Panel judges on "why this model" not just "we used model X" |
| 5 | **Create `requirements.txt`** | Anyone reproducing your work needs this |
| 6 | **Create `README.md`** | First thing anyone opens in the repo |
| 7 | **Create `docs/dataset_description.md`** | Dataset = foundation of everything |
| 8 | **Prepare viva answers** | Practice out loud, not just read |

---

*Last updated: 2026-06-02 | EPAIB Batch 05, Group 4, IIM Lucknow*
