# Presentation Q&A Preparation
## TRACE — Tiger Recognition & Automated Census Engine
## EPAIB Batch 05 | Group 4 | IIM Lucknow

> Every choice in this pipeline was deliberate.
> Know WHY you chose it, WHAT you rejected, and WHAT you would do differently with more data/time.

---

## SECTION A — MODEL CHOICES

---

### Q1. Why YOLOv8? Why not Faster R-CNN or SSD?

**Our choice:** YOLOv8n (pretrained on COCO)

**What we considered:**

| Detector | Type | Speed | Accuracy | Our verdict |
|---|---|---|---|---|
| **YOLOv8n** | One-stage | ~130 fps | Good | **Chosen** |
| Faster R-CNN | Two-stage | ~5–7 fps | Excellent | Too slow for 1TB scale |
| SSD | One-stage | ~60 fps | Moderate | Lower accuracy than YOLO |
| RetinaNet | One-stage | ~15 fps | Very good | Heavier than needed |

**Why YOLO specifically:**
- One-stage detector: detects all objects in a single forward pass — no separate region proposal step
- Real-time capable: at 130fps a ranger can review live camera feed, not just batch
- YOLOv8 (Ultralytics 2023) outperforms v5/v3 on small-object detection — critical when tiger is far from camera
- Production-grade Python API: `from ultralytics import YOLO` — clean integration

**Why not Faster R-CNN despite better accuracy:**
- Two-stage: first proposes regions, then classifies each — at 1TB = 4.5M images, ~5fps means 250 hours of compute time vs 9 hours with YOLO
- Conservation team needs results before patrol shifts, not next week

**Honest limitation:**
COCO has no tiger class. We use proxy classes 15 (cat), 16 (dog), 17 (horse), 24 (zebra) as stand-ins.
This is a known gap. Fine-tuning YOLOv8 on tiger-annotated data is Phase 2.

---

### Q2. Why ResNet50 for embeddings? Why not EfficientNetB3, VGG16, or InceptionV3?

**Our choice:** ResNet50 pretrained on ImageNet, classification head removed

**What we considered:**

| Model | Params | Embedding dim | Why rejected |
|---|---|---|---|
| **ResNet50** | 25M | 2048 | **Chosen** |
| VGG16 | 138M | 4096 | 5x more params, no skip connections, overfits easily |
| InceptionV3 | 24M | 2048 | Similar size, less community support, trickier to modify |
| EfficientNetB3 | 12M | 1536 | Smaller but needs fine-tuning for typed classification |
| ResNet18 | 11M | 512 | 512-dim too low for discriminating individual stripes |
| ResNet101 | 45M | 2048 | Same 2048 output, nearly 2x params, marginal gain |
| MobileNetV2 | 3.4M | 1280 | Built for mobile/edge, embedding less rich |

**Why ResNet50 specifically:**
- **Skip connections** (He et al., 2016): `output = F(x) + x`
  - Residual shortcut bypasses layers if they add no value
  - Solves vanishing gradient — gradients flow directly back through identity connections
  - Trains deeper networks without degradation
- **2048-dim embedding** is rich enough to discriminate individual tiger stripe patterns
- **ImageNet pretraining** on 1.2M images gives it strong visual feature understanding
- **Proven in metric learning**: ReID (person re-identification) research consistently uses ResNet50 as backbone

**Why we did NOT fine-tune it:**
We have no identity ground truth. Our annotation files contain only bounding boxes (x1,y1,x2,y2).
There are no Tiger_ID labels in the dataset. You cannot fine-tune a classifier without class labels.
ResNet50's pretrained features capture visual similarity well enough for stripe matching.

**The academic connection:**
Our approach mirrors DeepSORT (Wojke et al., 2017) which uses ResNet for appearance embedding
in multi-object tracking. Same principle: pretrained CNN features + distance metric for ReID.

---

### Q3. Why Cosine Similarity? Why not Euclidean distance?

**Our choice:** Cosine similarity with threshold 0.83

**The mathematical difference:**

```
Cosine similarity  = (A · B) / (|A| × |B|)   → measures ANGLE between vectors
Euclidean distance = √Σ(Aᵢ - Bᵢ)²            → measures ABSOLUTE DISTANCE
```

**Why cosine for tiger identity:**

Scenario: Tiger_003 photographed in daylight (bright) vs infrared night (dark).
- Daylight embedding: [0.8, 0.6, 0.7, ...] — large values, high magnitude
- Night embedding:    [0.4, 0.3, 0.35, ...] — same pattern, but half the scale

Euclidean distance sees these as FAR APART (different magnitudes).
Cosine similarity sees these as VERY CLOSE (same direction = same stripe pattern).

**Rule:** When the scale of vectors varies but the pattern is what matters → use cosine.

**Why not trained Siamese network:**
A Siamese network (contrastive loss or triplet loss) would learn a better metric, but requires:
- Positive pairs: (Tiger_003, img1), (Tiger_003, img2) — same individual, different photos
- Negative pairs: (Tiger_003, img1), (Tiger_007, img3) — different individuals

We don't have identity labels. No labels = no Siamese training.
Cosine on pretrained ResNet50 is the best we can do without ground truth.

**The 0.83 threshold — honest answer:**
It was set empirically, not calibrated from data. The right process is:
1. Manually annotate 30 val images with known Tiger IDs
2. Compute all pairwise cosine similarities (same-tiger and different-tiger pairs)
3. Plot ROC curve: FNR vs FPR for each threshold value
4. Choose threshold where FNR < 5% (our target)

This is an open issue. We know 0.83 is not validated — we acknowledge it in limitations.

---

### Q4. Why CLAHE? Why not standard histogram equalisation or PIL brightness boost?

**Our choice:** CLAHE on the L channel of LAB colour space

**What we considered:**

| Method | What it does | Problem |
|---|---|---|
| PIL brightness boost (Colab approach) | Multiplies all pixel values by factor | Linear — amplifies already-bright areas, crushes dark ones |
| Standard HE | Stretches histogram globally | Global — over-brightens background if one region is very bright |
| **CLAHE (our choice)** | Adaptive HE per tile, contrast limited | Local, contrast-controlled |

**Why CLAHE is better for camera traps:**
- Camera trap images have extreme local variation: tiger body (dark) vs sky/clearing (bright)
- Standard HE: brightens the whole image uniformly → washes out tiger stripes
- CLAHE: divides image into 8×8 tiles, equalises each independently
  - Dark tiger body gets contrast enhanced
  - Bright background is not over-amplified
- `clipLimit=2.0`: caps contrast amplification so noise is not amplified

**Why LAB colour space instead of BGR directly:**
LAB separates luminance (L) from colour (A=green-red, B=blue-yellow).
- We enhance ONLY L (lightness) — preserves the original orange/stripe colours
- If we applied CLAHE to all 3 BGR channels independently, we'd introduce colour artefacts
- The stripe colour is a key feature for morph classification — we must not corrupt it

---

### Q5. Why Dark Channel Prior dehazing? Why not skip it?

**Our choice:** He et al. (2010) Dark Channel Prior

**Why it's needed:**
Sundarbans is a monsoon-climate mangrove forest.
- Seasonal fog (Nov–Jan)
- Rain haze (Jun–Sep)
- Dust scatter in dry season

A foggy image looks like this to YOLO: one bright blob where a tiger should be distinct.
DCP removes the haze layer and recovers scene radiance — the tiger becomes detectable.

**Why not a learned dehazing network (AOD-Net, etc.):**
- Learned dehazing needs haze/clean image pairs for training
- We don't have paired camera trap images
- DCP works with zero training data — physics-based, not learned
- Good enough for our use case (not a research contribution on dehazing)

**The physics:**
`I(x) = J(x)·t(x) + A·(1 - t(x))`
- I = what the camera sees (hazy)
- J = the real scene (what we want)
- t = transmission map (how much light gets through)
- A = atmospheric light (brightness of the haze itself)

DCP estimates t from the dark channel and recovers J.

**Academic note:** He et al. won the IEEE CVPR 2009 Best Paper award for this work.

---

### Q6. Why CenterCrop(224) and not Resize(224, 224)?

**Our choice:** `transforms.Resize(256)` → `transforms.CenterCrop(224)`

**The problem with Resize(224, 224):**
A tiger photographed in landscape (wide) format: 800×400 pixels.
Resize(224, 224) squashes it to a square — the tiger becomes short and fat.
The stripe pattern is geometrically distorted.

ResNet50 was pretrained on ImageNet with CenterCrop. ImageNet images are natural,
varied-aspect-ratio photos. The model learned from uncropped proportions.

**Why CenterCrop works:**
1. `Resize(256)` — scales the shorter edge to 256, preserving aspect ratio
2. `CenterCrop(224)` — takes a 224×224 crop from the centre
3. The tiger body (usually centred in the YOLO crop) is preserved at correct proportions

**The stripe pattern argument:**
Bengal tiger stripes are the identity fingerprint.
Aspect ratio distortion changes the apparent stripe width and spacing.
The same tiger with distorted stripes could drop below the 0.83 cosine threshold
and be re-enrolled as a new individual — inflating the tiger count.

---

### Q7. Why running average embedding? Why not just replace or store all embeddings?

**Our choice:** Running average: `new_db = (old × n + new) / (n + 1)`

**Three options considered:**

| Option | What happens | Problem |
|---|---|---|
| Replace | Each sighting overwrites the DB | One bad (partial, blurry) sighting corrupts the fingerprint |
| Store all, average at query time | Keep every embedding, average when matching | O(n) memory per tiger, slows as sightings grow |
| **Running average (our choice)** | DB is always a single vector, updated incrementally | Memory = constant (2048 floats per tiger regardless of sightings) |

**Why running average is mathematically sound:**
After 5 sightings: DB = (E1 + E2 + E3 + E4 + E5) / 5
After 6th sighting: DB = (DB × 5 + E6) / 6 = (E1+E2+E3+E4+E5+E6) / 6 ✓

The running average is identical to the full average without storing all embeddings.
Each new full-body sighting makes the DB embedding more reliable.

**Why partial crops must NOT update the DB:**
A tail crop produces a 2048-dim embedding biased toward tail features, not stripes.
If used to update the DB, it shifts the running average away from the true fingerprint.
Next time the full tiger appears, cosine similarity drops → false new tiger enrollment.

---

### Q8. Why HSV for colour morph? Why not EfficientNetB3?

**Our choice:** Rule-based HSV thresholds

**The EfficientNetB3 problem:**
EfficientNetB3 needs labelled training data:
```
orange_tiger/   ← needs 100+ labelled images
white_tiger/    ← needs 100+ labelled images
black_tiger/    ← needs 100+ labelled images
golden_tiger/   ← needs 100+ labelled images
snow_white/     ← needs 100+ labelled images
```
We don't have morph-labelled images. HSV rules work with zero labels.

**Why HSV works for colour morph:**
Tiger morphs are defined by colour:
- White tiger: high Value (bright), very low Saturation (no orange)
- Black tiger: very low Value (dark overall)
- Golden tiger: Hue in 25–40 range (yellow-orange), lower Saturation
- Snow White: maximum brightness + near-zero Saturation + near-zero variance

HSV directly encodes these properties. It is the natural representation for colour-based rules.

**The honest trade-off:**
HSV rules are interpretable but not robust to unusual lighting.
A white tiger photographed in red sunset light could be misclassified as orange.
A trained EfficientNetB3 would learn to handle these edge cases.
This is Phase 2 work — needs labelled morph data first.

---

### Q9. Why pretrained YOLOv8n and not fine-tuned?

**Honest answer:** We don't have YOLO-format annotated training data for tigers.

YOLO fine-tuning needs:
- Images in YOLO format: one `.txt` file per image
- Each line: `class_id x_centre y_centre width height` (normalised 0–1)
- Roboflow-style dataset with train/val/test splits

Our annotation files (`train_annotations.csv`) contain only bounding box coordinates (x1,y1,x2,y2).
They are not in YOLO format. They do not have class IDs (no tiger class exists in COCO anyway).

**What we would need for fine-tuning:**
1. 500+ tiger images with bounding boxes annotated in Roboflow
2. Export in YOLOv8 format
3. Training run: `model.train(data='tiger.yaml', epochs=100)`

**The proxy class approach:**
COCO class 15 (cat), 16 (dog), 17 (horse), 24 (zebra) are visually closest to tigers
among the 80 COCO classes. These are not perfect — they will produce false positives
(actual dogs detected) and false negatives (tigers missed). But they work well enough
for a prototype.

---

## SECTION B — ARCHITECTURE DECISIONS

---

### Q10. Why 3 separate models instead of one end-to-end model?

**The pipeline:** Triage → Detection+Identity → Classification

**Why not one model:**
An end-to-end model that does triage + detection + identity + morph in one forward pass
would require thousands of labelled examples across all tasks simultaneously.
We have no such labelled dataset.

**The engineering principle: cheapest filter first**
- Triage (rule-based): costs microseconds, rejects ~15% of images before any ML runs
- Detection (YOLO): costs ~10ms per image, applied only to images that passed triage
- Identity (ResNet50): costs ~50ms per crop, applied only to detected tiger crops
- Classification (HSV): costs microseconds, applied only to valid detections

If we ran ResNet50 on every image (including blurry/overexposed ones), we'd waste 15% of compute.
At 4.5M images/day scale, 15% compute waste = significant cost and time.

**Industry parallel (from Prof. Mahesh Balan):**
At PayPal, fraud detection uses exactly this pattern:
1. Rule-based filter: flag transactions with obvious signals (amount > $10,000, new device, foreign IP)
2. ML model: run only on transactions that pass the rule filter
3. Deep model: run only on high-risk transactions flagged by ML

---

### Q11. Why in-memory DB with JSON persistence? Why not SQLite or FAISS?

**Current: Python dict + JSON file**

**Why it works for now:**
13 tigers × 2048 floats × 4 bytes = ~106KB total DB size. JSON is fine.
Linear scan of 13 embeddings per detected crop = 13 cosine similarity calculations.
At this scale, a for-loop is faster than FAISS setup overhead.

**Why FAISS for production:**
At India-wide scale: 3,000+ individual tigers across 50 reserves.
- 3,000 × 2048 dim embeddings = ~25MB index
- Linear scan: O(n) = 3,000 comparisons per crop per image per camera
- FAISS approximate nearest neighbour: O(log n) — significantly faster at scale
- FAISS supports GPU acceleration

**Why not SQLite:**
SQLite is for structured data. Storing 2048-float embeddings in SQLite rows
requires serialisation/deserialisation overhead. JSON is simpler for this use case.
For production with metadata (location, date, camera_id), SQLite is the right choice.

**This is a known limitation** — acknowledged in architecture.md and Slide 12.

---

### Q12. What is the False Negative Rate? Have you measured it?

**Honest answer:** We have not measured the true FNR because we have no ground truth.

**What we know:**
- Target: FNR < 5% (stated in problem_statement.md)
- From notebook 04 (simulated): 3.9% FNR — but this is on simulated data, not labelled val
- Devil's Advocacy: two independent runs produce consistent results (13 tigers both times)
  — this proves reproducibility, NOT accuracy

**What FNR means here:**
FNR = False Negative Rate = missed tigers / total tigers present

In conservation: missing a tiger is worse than a false alarm.
A poaching event detected by a false alarm wastes ranger time.
A missed tiger in the census leads to wrong conservation policy decisions.
"Missing a tiger is a data point lost for conservation."

**How to compute the real FNR:**
1. Manually annotate 30 val images: record which tiger is in each image (ground truth)
2. Run the pipeline on those 30 images
3. Compare pipeline output to ground truth
4. FNR = images where pipeline said "no tiger" but ground truth had a tiger

This is the highest-priority open issue before deployment.

---

### Q13. Why 187 training images? That seems small.

**Why the dataset is small:**
This is a publicly sourced prototype dataset (Google Images / iNaturalist).
The West Bengal Wildlife Department has ~1TB of historical camera trap data
but access requires regulatory permissions and field partnerships.
Mr. Anamitra Lahiri (group member) is liaising with the Programme Director.

**What we do with 187 images:**
- Pretrained models don't need your data for training
- YOLOv8n and ResNet50 are trained on ImageNet (1.2M images) and COCO (120K images)
- Our 187 images are purely for evaluation and demonstration — not for training

**Why it still works:**
- Image triage: rule-based, no data needed
- Detection: pretrained YOLO, no fine-tuning
- Identity: pretrained ResNet50 features, no fine-tuning
- Colour morph: HSV rules, no data needed
- Only thing that would benefit from more data: fine-tuning YOLO and ResNet50

---

## SECTION C — EVALUATION QUESTIONS

---

### Q14. You have 13 unique tigers. How confident are you in that number?

**Honest answer:** Moderately confident in consistency, less confident in absolute accuracy.

**What we can prove:**
- Run 1: 13 unique tigers | Run 2: 13 unique tigers → CONSISTENT (devil's advocacy)
- This means the pipeline is deterministic and reproducible
- Consistency ≠ correctness

**What could make it wrong:**
- Threshold 0.83 too high → splits same tiger into two identities (overcounts)
- Threshold 0.83 too low → merges different tigers into one identity (undercounts)
- Proxy YOLO classes miss some tiger crops → misses tigers entirely
- ResNet50 pretrained features not optimal for stripe discrimination

**The right answer at viva:**
"Our devil's advocacy proves the system is consistent. Calibrating 0.83 against
labelled validation data is the next step before we can claim accuracy. We know
the number is trustworthy once the threshold is validated."

---

### Q15. How is TRACE different from just running YOLO?

Running YOLO on 187 images would give you:
- N detections of tiger-like animals
- No names, no identities, no census count

TRACE adds:
1. **Individual identity**: Tiger_003 in image 12 = same as Tiger_003 in image 89
2. **Census count**: 187 images, 13 UNIQUE individuals (not 187 separate "tigers")
3. **DB that persists**: run today, run next week, same Tiger_IDs are maintained
4. **Partial body protection**: a tail in one image doesn't create a false new tiger
5. **Quality filtering**: 15% of images are rejected before ML runs → cleaner results
6. **Explainability**: Grad-CAM shows WHERE the model is looking, not just WHAT it decided

"YOLO tells you a tiger is present. TRACE tells you WHICH tiger, how many times
it has been seen, and whether it is the same one from last month."

---

## SECTION D — PROFESSOR-SPECIFIC QUESTIONS

---

### Prof. Mahesh Balan (Deep Learning) will ask:

- "Walk me through layer by layer what ResNet50 does to a tiger crop."
  → Conv → BN → ReLU → [4 residual blocks] → GlobalAvgPool → 2048-dim vector
  → Each residual block: 3 conv layers + skip connection (`F(x) + x`)
  → The skip connection means gradients flow directly back without vanishing

- "Show me the Grad-CAM. What is it looking at?"
  → `reports/gradcam_outputs/` — red area should be on body/stripes, not background

- "What is your FNR? How did you compute it?"
  → 3.9% on simulated data from notebook 04. True FNR needs 30 annotated val images.

---

### Prof. Sowmya (Faculty Supervisor) will ask:

- "You said 13 tigers. Can you prove it?"
  → Consistent across 2 runs. Cannot prove accuracy without ground truth labels.

- "What is the subspecies classifier doing?"
  → Laplacian variance proxy — it is a placeholder. Do NOT present it as a result.

- "Why does the count say 13 when Sundarbans has 96+ tigers?"
  → 187 training images, sampled dataset, not all tigers present. Prototype only.

---

### Prof. Laxminarayanan G (AI Strategy) will ask:

- "What is the business case? What does the ROI look like?"
  → Manual review: 3 years for 1TB. TRACE: 7.4 hours. 92% effort reduction.
  → Rangers freed from desk review can spend time on field patrol.

- "How would you scale this to all 50 tiger reserves?"
  → FAISS for vector search, SQLite for metadata, REST API for field deployment,
  → Per-reserve DB instances, centralised census aggregation dashboard.

---

### Prof. Prakash Attili (Agile/IT) will ask:

- "What is your sprint plan? What is in scope vs out of scope?"
  → In scope: detection, identity, morph. Out of scope: real-time camera integration, mobile.
  → Two-week sprint cycles, MVP first (working pipeline), then iterate.

---

### Prof. Sumit Singh (GenAI/Product) will ask:

- "How would you make this accessible to a non-technical forest ranger?"
  → Gradio web interface: drag and drop image, see bounding boxes + tiger ID instantly.
  → No command line needed. One-click population report export.

- "Where is the demo?"
  → `py -3 src/tiger_ai_complete.py --image tiger_10.png` — annotated image in `reports/annotated_images/`

---

## QUICK REFERENCE — ONE-LINE ANSWERS

| Decision | One-line answer |
|---|---|
| YOLOv8 over Faster R-CNN | Speed at scale: 130fps vs 5fps — 1TB can't wait 250 hours |
| ResNet50 over EfficientNetB3 | 2048-dim rich embeddings, proven in ReID, no fine-tuning needed |
| Cosine over Euclidean | Scale-invariant — same tiger at day vs night has same direction, different magnitude |
| CLAHE over PIL brightness | Adaptive local enhancement, preserves colour channels |
| DCP dehazing | Physics-based, works without training data, Sundarbans needs it |
| CenterCrop over Resize | Preserves aspect ratio — stripe distortion = false identity mismatch |
| Running average over replace | One bad sighting doesn't corrupt the fingerprint |
| HSV over EfficientNetB3 | No morph labels exist — HSV is zero-data, interpretable |
| 3 models over 1 end-to-end | Cheapest filter first — 15% compute saved before any ML runs |
| Pretrained not fine-tuned | No identity ground truth labels in dataset |
| JSON over SQLite | 13 tigers = 106KB — JSON is simpler at this scale |
| 0.83 threshold | Empirically set — known limitation, needs ROC curve calibration |

---

*Last updated: 2026-05-31 | Group 4 | EPAIB Batch 05 | IIM Lucknow*
*Use this document for viva prep, Sowmya mam review, and panel presentations.*
