# What We Built and What We Fixed
## EPAIB Batch 05 | Group 4 | IIM Lucknow | Tiger Enumeration Project

---

## The Starting Point — Two Scripts, Two Problems

We had two scripts at the beginning of this work:

### Script 1 — The Colab Script (`copy_of_tiger_ai_system_complete.py`)
This was a Google Colab notebook that detected tigers and classified them by colour
(orange, white, golden, black, snow white). It was well-documented and visually polished.

**But it had serious problems:**
- It only ran inside Google Colab — crashed immediately on any normal laptop
- It needed a GPU — without one, line 64 crashed the whole program
- It needed 90 minutes of training before detecting even one tiger
- Most critically: it **counted sightings, not unique individuals**
  - If Tiger_007 appeared in 20 images, it reported 20 tigers, not 1
  - This is not a census — it is a sighting log
- It had a hidden bug: YOLO was looking at the original foggy image,
  not the cleaned one (line 1030 — the preprocessing was being ignored)

### Script 2 — The TRACE Pipeline (`tiger_pipeline.py`)
This was our own locally-built pipeline. It ran on any laptop, had no training requirement,
and could identify individual tigers using stripe-pattern fingerprints.

**But it was missing some things from the Colab script:**
- No coloured bounding boxes drawn on the images
- No crop padding (tigers at the edge of their detection box looked cut off)
- YOLO had the same preprocessing bug (looking at original image, not cleaned one)
- No visibility label in the output (Full Body vs Partial Body)

---

## What We Did — The Full Work Done This Session

### Step 1 — We Compared Both Scripts Side by Side

We read through every line of both scripts and built a complete comparison table
showing what each script could do, what it was missing, and which approach was better
for each feature. This comparison is saved in:
`docs/learning_notes_colab_vs_trace.md`

---

### Step 2 — We Fixed the TRACE Pipeline (v4 → v5)

We updated `src/tiger_pipeline.py` with the things the Colab script did better:

| What We Added | Why |
|---|---|
| Coloured bounding boxes on output images | Visual proof of what the model detected |
| 15px padding around each detected crop | Prevents edges being cut off when feeding into ResNet50 |
| Visibility label in CSV (Full_Body / Partial_Body) | Shows the forest officer how much of the tiger was visible |
| YOLO bug fix — now receives cleaned image array | YOLO was ignoring all preprocessing — now it sees the fog-removed version |
| Annotated images saved to `reports/annotated_images/` | Each image saved with tiger IDs and colour boxes drawn on it |

---

### Step 3 — We Created a New Complete Script (`tiger_ai_complete.py`)

We built a brand new script that combines the best of both worlds:

**From the Colab script we kept:**
- The `analyze_image()` function — clean, single-image entry point
- Bounding box visualisation with morph-specific colours
- Crop padding to avoid edge artefacts
- Population report format

**From TRACE we kept (and the Colab script was missing these):**
- Individual tiger identity (Tiger_001, Tiger_002...)
- Persistent JSON database that remembers tigers across sessions
- CLAHE preprocessing (local contrast enhancement for dark images)
- Dark Channel Prior dehazing (fog/rain/haze removal)
- IR/night image detection and handling
- Image triage with skip logic (blurry/overexposed/underexposed images skipped)
- Partial body DB protection (a tail crop won't corrupt Tiger_003's fingerprint)
- Grad-CAM explainability (--gradcam flag shows where model looked)
- Visibility label in CSV

**New things we added that neither script had:**
| New Feature | What It Does | Why It Matters |
|---|---|---|
| Shadow Detection (`is_shadow_crop`) | Checks if a detected crop is dark and colourless — if yes, rejects it as a shadow | Tiger shadows look like tigers. Without this, one tiger = two counted. |
| Corner Trace Detection (`is_corner_trace`) | Checks if a detection box is at the image border | Partial tigers entering the frame from the side should not update the identity database |
| Overlap Guard (`check_box_overlap`) | Rejects a new box if it overlaps an already-accepted box by more than 45% | Shadow boxes that pass the colour test but are spatially near the real tiger are caught here |
| Corner_Trace visibility label | Third label added alongside Full_Body and Partial_Body | Distinguishes "entered frame partially" from just "small body" |

---

### Step 4 — We Prepared for the Viva (Q&A Document)

We created `docs/presentation_qa_prep.md` — a complete document answering every
likely question from each professor:

- Why did you use YOLOv8 and not another detector?
- Why ResNet50 and not EfficientNetB3?
- Why cosine similarity and not Euclidean distance?
- What were the alternatives you considered and rejected?
- Why CLAHE instead of standard brightness adjustment?
- What is your False Negative Rate?
- Why 0.83 as the cosine threshold?

Each answer includes: our choice, the alternatives, why we chose what we chose,
and an honest statement of the limitation.

---

### Step 5 — We Rebuilt the Presentation (12 Slides)

We rewrote `presentation/build_ppt.py` from 22 slides down to exactly 12.
Key changes:
- India-wide scope (not just Sundarbans — 50+ reserves, 4.5M images/day)
- New Slide 06: "Why These Models — Choices and Alternatives" table
- Slide 03 Camera Trap Challenges now includes: tiger shadow detection,
  multiple tigers in one frame, partial tigers at image corners
- Slide 12 Limitations section is honest about what is still open

---

### Step 6 — We Ran the Professor Panel Analysis

We analysed the Colab script through the lens of all 5 professors:

| Professor | Their Focus | Key Finding |
|---|---|---|
| Prof-MB (Deep Learning) | Architecture and model choices | YOLO bug, EfficientNetB3 wrong for census, no Grad-CAM, no identity |
| Prof-SS (AI Products) | Can it be deployed and used? | 3 crash points, no CLI, good analyze_image() pattern, Colab is a prototype |
| Prof-LG (AI Strategy) | Business outcome and enterprise use | Colab fails all 5 enterprise criteria, no audit trail, no persistent DB |
| Prof-SW (Validation) | How was accuracy measured? | Evaluated on wrong data, wrong metric (colour type not identity), FNR unmeasured |
| Prof-PR (Presentation) | Can it be presented and defended? | Cannot live-demo Colab, TRACE can, training graphs are presentation assets |

---

### Step 7 — We Created the Panel Feedback Excel

`docs/panel_feedback_report.xlsx` — 4 sheets:

| Sheet | Contents |
|---|---|
| All Feedback | All 31 feedback items in one place — plain English, what Colab does, what TRACE does, severity |
| By Professor | Same data grouped by professor with their colour coding |
| Viva Prep | 12 Q&A pairs — the exact question each professor will ask + your plain-English answer |
| Action Tracker | 19 action items: 12 done, 3 open, 4 Phase 2 roadmap |

---

## Where Everything Was Saved

```
epaib-batch05-group4/
├── src/
│   ├── tiger_pipeline.py          ← Updated to v5 (bounding boxes, bug fix)
│   └── tiger_ai_complete.py       ← New complete script (best of both + new features)
├── presentation/
│   ├── build_ppt.py               ← Rebuilt to 12 slides, India-wide scope
│   └── TRACE_Presentation.pptx   ← Generated 12-slide presentation
├── docs/
│   ├── learning_notes_colab_vs_trace.md   ← Full comparison of both scripts
│   ├── presentation_qa_prep.md            ← Viva Q&A for all professor topics
│   ├── panel_feedback_report.xlsx         ← This Excel (plain English feedback)
│   └── project_work_summary.md            ← This document
└── agents/
    └── professor_panel.py         ← Multi-professor agent system (Anthropic SDK)
```

---

## What Is Still Open (Honest Assessment)

| Open Item | Why Not Done Yet | What Is Needed |
|---|---|---|
| True False Negative Rate | Requires images with confirmed tiger counts | 30 annotated images from WII or Project Tiger |
| Calibrate the 0.83 threshold | Requires ROC curve on labelled data | Same annotated images |
| Add charts to TRACE report | A 1-day task — not yet scheduled | Just Python time |
| Add threshold tuning helper | Borrowed from Colab Cell 5.3 — not ported yet | 1-day task |
| Streamlit browser UI | Phase 2 product step | Build on top of existing TRACE backend |

---

*Last updated: 2026-05-31 | EPAIB Batch 05 | Group 4 | IIM Lucknow*
