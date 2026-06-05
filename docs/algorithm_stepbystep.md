# TRACE Pipeline — Step-by-Step Algorithm
## Tiger AI Ultimate v3.0 | EPAIB Batch 05 | Group 4 | IIM Lucknow

---

## The 7 Core Objectives — What This Pipeline Must Deliver

These are the primary questions the TRACE pipeline answers for every batch of camera trap images:

| Step | Question | Pipeline Stage | Output |
|------|----------|---------------|--------|
| **1** | How many images can be used vs cannot be used? | Stage 2 — Triage | Usable count + per-image rejection reason |
| **2** | Can unusable images be recovered by enhancement? | Stage 2b — Enhancement retry | Enhanced image re-checked; use if passes |
| **3** | Does this image contain a tiger? | Stage 4 + Stage 5 | Yes / No per image |
| **4** | How many tigers are in this image? | Stage 6 — Grid Scan | 1, 2, or more per image |
| **5** | How many total tiger detections across the dataset? | Census summary | Total detection count |
| **6** | How many **unique** tigers have been seen? (population estimate) | Stage 8 — Identity matching | Unique Tiger_ID count |
| **7** | What type/colour morph is each tiger? | Stage 7 — Colour morph | White / Black / Orange / Golden / Snow White |

---

### Step 1 — Image Usability Filter

For a batch of 500 images, the pipeline first sorts them into:
- **Usable** — Clean daytime images, IR night images (automatically processed differently)
- **Unusable** — Blurry, overexposed, underexposed

Every rejected image gets a reason logged in the CSV output.

---

### Step 2 — Enhancement Retry for Borderline Images

Images that initially fail the quality gate are not simply discarded. Before final rejection, the pipeline attempts:
1. **Gamma correction** — auto-lifts dark images toward target brightness 80
2. **CLAHE with stronger clip limit** (8.0 vs standard 2.0) — recovers contrast in shadow-heavy crops
3. **Re-check** — if the enhanced version now passes blur/brightness thresholds, it is used; otherwise it is logged as truly unusable

This means the pipeline gives every image two chances before giving up.

---

### Step 3 — Tiger Detection (Yes / No)

Each usable image goes through a **3-stage cascade detector** designed so no tiger is missed:
- Stage 1 (OIV7): YOLOv8 with a direct Tiger class — most accurate
- Stage 2 (COCO proxy): Falls back to cat/dog/horse/zebra shape proxies
- Stage 3 (Whole-image): ResNet50 judges the entire frame — last resort

An image is marked "has tiger" only if the detection + species verification gate both confirm.

---

### Step 4 — Count Tigers Per Image (Multi-Tiger Detection)

One image can contain more than one tiger. The pipeline detects:
- **Multiple YOLO boxes** — when YOLO draws separate boxes for each tiger
- **Grid scan** — when YOLO finds no separate boxes but the frame has side-by-side tigers: splits the frame left/right and top/bottom, verifies each half independently
- **Over-counting guard** — fingerprint similarity check prevents a single large tiger straddling both halves from being double-counted (threshold 0.68)

---

### Step 5 — Total Detections Across Dataset

After all images are processed, the CSV report totals:
- Images processed
- Images with tigers
- Images without tigers (no tiger detected or species-rejected)
- Images skipped (unusable)
- Total tiger detections (sum across all images)

---

### Step 6 — Unique Tiger Count (Population Estimate)

The pipeline distinguishes between "this tiger was photographed 40 times" and "40 different tigers were photographed". For each detection:
1. Extract a 2048-dimensional stripe fingerprint (ResNet50 layer4)
2. Compare against every known tiger's fingerprint using cosine similarity
3. Score ≥ 0.83 → same tiger seen again (update its running average profile)
4. Score < 0.83 → new individual, enrol as Tiger_001, Tiger_002, etc.

**ViewPoint-aware census** (prevents double-counting):
- A tiger seen from the left and from the right has different-looking stripes
- Conservative estimate = `max(left-unique, right-unique, frontal-unique)`
- Liberal estimate = union of all IDs

---

### Step 7 — Tiger Type Classification

Each detected tiger is classified into one of five colour morphs based on HSV analysis:

| Morph | Visual | Detection Logic |
|-------|--------|-----------------|
| **Orange Standard** | Standard Bengal orange + black stripes | Default category |
| **White** | White body, faint stripes | High brightness, very low saturation |
| **Snow White** | Pure white, near-invisible stripes | Very high brightness, near-zero saturation, uniform |
| **Black (Pseudomelanistic)** | Very dark body, faint stripes | Very low mean value (V channel) |
| **Golden** | Golden-yellow coat, faint stripes | Yellow-orange hue (H 25–40°), low-medium saturation |

This classification runs on every confirmed detection at zero extra cost (HSV analysis is fast, no extra model needed).

---

## Pipeline Overview

The pipeline takes one camera trap image and produces:
- Tiger detected? Yes / No
- If yes: Which individual tiger is it?
- How many unique tigers have been seen across all images?

There are **9 stages** executed in sequence for every image.

---

## Stage 1 — Image Load

```
Input: File path to image (JPG / PNG / JPEG / WEBP)
```

1. Try `cv2.imread()` — OpenCV native loader
2. If that returns None (HEIC, unusual format): fall back to `PIL.Image.open()` converted to BGR
3. If both fail: return empty result, skip image
4. Store the raw unprocessed image as `img` — needed later for dual-scan detection

---

## Stage 2 — Image Triage (Quality Gate)

```
Purpose: Reject images too degraded to extract stripe features
```

**Step 2.1 — IR / Night detection (runs FIRST)**
- Convert to HSV colour space
- Calculate mean saturation across the image
- If `mean_saturation < 15` AND `std_saturation < 10` → image is greyscale IR (night camera)
- Assign bucket: `IR_Night`
- IR images bypass all brightness and blur rejection — they are valid despite being dark and desaturated

**Step 2.2 — Brightness check (only if NOT IR)**
- Convert to greyscale, compute `mean_brightness`
- If `mean_brightness > 240` → `Unusable_Overexposed` → SKIP
- If `mean_brightness < 8` → `Unusable_Underexposed` → SKIP
- *Threshold 8 (not 20): camera trap images can be legitimately dark at brightness 11-13*

**Step 2.3 — Blur check (only if NOT IR)**
- Apply Laplacian filter to greyscale image
- Compute variance of Laplacian response
- If `variance < 25` → `Unusable_Blurry` → SKIP
- *Threshold 25 (not 80): camera trap images through foliage score 38-48 and are visually sharp*

**Step 2.4 — Result**
- `Clean` → proceed to preprocessing
- `IR_Night` → proceed to preprocessing (special path)
- `Unusable_*` → log reason, skip image entirely

> **Why IR check runs first (Bug Fix 2):** If brightness check ran first, dark IR images (brightness=11) would be rejected as "underexposed" before the pipeline even identified them as valid night shots. IR check must be the first gate.

---

## Stage 3 — Preprocessing (Image Enhancement)

```
Purpose: Enhance stripe contrast so ResNet50 and YOLO can see features clearly
```

**Path A — IR Night images:**
1. Convert BGR → greyscale (single channel)
2. Apply CLAHE (clipLimit=2.0) to boost local contrast of stripes
3. Convert back to BGR by stacking 3 identical channels
4. *Result: greyscale night image with enhanced stripe edges*

**Path B — Clean daytime images:**
1. **Gamma correction** (if mean brightness < 40):
   - Compute gamma: `γ = log(current_brightness/255) / log(80/255)`
   - Apply LUT-based gamma correction to lift image to target brightness ~80
   - *Purpose: dark daylight images become processable without overexposing bright areas*
2. **CLAHE** (clipLimit=4.0, tileGridSize=8×8):
   - Operates in LAB colour space on the L (luminance) channel only
   - Boosts local contrast of stripe patterns without changing colour
3. **Dark Channel Prior dehazing** (for Sundarbans monsoon haze):
   - Estimates atmospheric veil in the image
   - Subtracts haze layer to recover stripe detail washed out by fog/mist

**Output:** `img_proc` — the enhanced image ready for YOLO detection

---

## Stage 4 — Cascade Detection (3-Stage YOLO)

```
Purpose: Draw bounding boxes around every tiger in the frame
Input: img_proc (preprocessed) AND img (raw original) — dual scan
```

**Stage 4.1 — OIV7 Direct Tiger Detection**
- Run `YOLOv8n` with Open Images V7 weights on `img_proc`
- Open Images V7 has a direct "Tiger" class — no proxy needed
- Confidence threshold: 0.09 (low because species gate provides secondary verification)
- Collect all boxes where `class_name contains "tiger"`
- **Dual scan:** Also run OIV7 on the raw `img` (original, unprocessed)
  - CLAHE + DCP preprocessing can suppress local contrast features that YOLO uses
  - Example: 15.jpeg — OIV7 finds 2 tiger boxes on raw, only 1 on preprocessed
  - Merge boxes from both scans using **IoU deduplication** (IoU threshold 0.30)
  - Only add raw-image boxes that don't overlap with a preprocessed detection
- If any boxes found → go to Stage 5, skip Stage 4.2 and 4.3

**Stage 4.2 — COCO Proxy Detection (if Stage 4.1 finds nothing)**
- Run `YOLOv8n` with COCO weights on `img_proc`
- COCO has NO Tiger class — use shape proxies instead:
  - Class 15 = cat (feline body shape)
  - Class 16 = dog (quadruped silhouette)
  - Class 17 = horse (large quadruped, similar side profile)
  - Class 22 = **zebra** (stripe pattern — BEST proxy; tiger stripes trigger zebra detector)
  - *Note: class 22 = zebra, NOT class 24 (which is backpack — fixed in Bug Fix 7)*
- Confidence threshold: 0.35 (higher because proxy classes ≠ tiger)
- **Dual scan:** Also run COCO on raw `img` if preprocessed COCO finds nothing
- All proxy boxes go to Stage 5 for species verification
- If any boxes found → go to Stage 5, skip Stage 4.3

**Stage 4.3 — Whole-Image Fallback (if both Stage 4.1 and 4.2 find nothing)**
- Use the entire image as a single bounding box: `(0, 0, width, height)`
- Common case: close-up tiger fills entire frame → YOLO sees "no tiger within scene"
- Confidence = 0.0 (no YOLO confidence — ResNet is the only gate)
- Mandatory species verification in Stage 5

---

## Stage 5 — Species Verification Gate (ResNet50)

```
Purpose: Confirm that each bounding box actually contains a tiger
Applied to: COCO proxy boxes + Whole-image fallback
Skipped for: OIV7 boxes (OIV7 already confirmed Tiger class)
```

**For each candidate crop:**

1. Extract crop from `img_proc` using box coordinates + 10px padding
2. Reject if crop < 20×20 pixels (too small for ResNet)
3. Skip shadow guard for IR_Night images (IR = zero saturation = always looks like shadow)
4. Apply shadow guard for daytime: reject achromatic dark crops (mean_saturation < 30)

**Three-attempt strategy:**

**Attempt 1 — Standard top-3:**
- Resize crop to 224×224, normalise with ImageNet mean/std
- Run ResNet50 classification head (1000 ImageNet classes)
- Get top-3 predicted class labels
- If any label contains "tiger" → CONFIRMED ✓

**Leopard rejection check (between Attempt 1 and 2):**
- Count how many of top-3 are non-tiger big cats: leopard, snow leopard, jaguar, cheetah, cougar, lion, puma, panther, ocelot
- If 2 or more → REJECTED immediately — do NOT widen to top-5
- *Purpose: prevents "tiger cat" (a small South American wild cat) from accepting a leopard image*

**Attempt 2 — Brightened top-3 (for dark crops):**
- If `mean_brightness_of_crop < 60`: apply gamma correction + strong CLAHE to crop
- Re-run ResNet on brightened crop
- If any label contains "tiger" → CONFIRMED ✓
- Apply leopard rejection check again

**Attempt 3 — Top-5 widened (last resort):**
- Get top-5 predicted labels (safe only after leopard check passed)
- If any label contains "tiger" → CONFIRMED ✓

If all three attempts fail → REJECTED — not a tiger

---

## Stage 6 — Grid Scan (Multi-Tiger Detection)

```
Purpose: Detect multiple tigers in a single frame when YOLO finds no individual boxes
Triggered only when: Whole-image fallback confirmed a tiger in Stage 5
```

**Step 6.1 — Left / Right split:**
1. Split image into left half (`img[:, :mid_x]`) and right half (`img[:, mid_x:]`)
2. Run ResNet species gate independently on each half
3. If BOTH halves confirm tiger → extract fingerprints from each half, compute cosine similarity
4. **Over-counting guard:** if `similarity ≥ 0.68` → same tiger spanning the frame → keep 1 detection, set `grid_handled = True`
5. If `similarity < 0.68` → genuinely 2 different animals → replace whole-frame with 2 half-frame detections, set `grid_handled = True`

> Calibrated values: single tiger half_sim ≈ 0.72 (caught); mother+cub half_sim ≈ 0.39 (correctly split)

**Step 6.2 — Top / Bottom split (`if not grid_handled` only):**
- This block is SKIPPED if left/right already examined both halves (flag=True)
- WHY: a single tiger also makes top (head/back) and bottom (belly/legs) both pass the species gate, but with sim≈0.44 < 0.68. Without the flag, this creates 2 spurious detections from 1 tiger.
1. Split image into top half (`img[:mid_y, :]`) and bottom half (`img[mid_y:, :]`)
2. Run ResNet species gate independently on each half
3. Apply same fingerprint similarity guard (threshold 0.68)
4. If genuinely 2 tigers → 2 half-frame detections

**Step 6.3 — Fallback:**
- If no split produces 2 confirmations, or one half fails species gate → keep single whole-frame detection

> **Example:** Mother + cub walking side by side → left half = adult tiger, right half = cub → half_sim ≈ 0.39 < 0.68 → 2 fingerprints enrolled correctly.

---

## Stage 7 — ViewPoint Classification

```
Purpose: Classify each detection as Left_Flank / Right_Flank / Frontal
Used for: ViewPoint-aware census (avoids counting left and right flank as 2 tigers)
```

**Logic:**
- Compute box centre relative to image width: `cx = (x1 + x2) / 2`
- If `cx < 35%` of image width → the tiger entered from the left → `Left_Flank`
- If `cx > 65%` of image width → the tiger exited to the right → `Right_Flank`
- Otherwise → `Frontal`

**Census rule:**
- Conservative estimate = `max(Left_count, Right_count, Frontal_count)`
- Rationale: same tiger photographed from left and right = 2 detections but 1 individual

---

## Stage 8 — Fingerprinting + Identity Matching

```
Purpose: Determine if this tiger has been seen before
Input: the confirmed crop
Output: Tiger_ID (existing or new) + match score
```

**Step 8.1 — Feature extraction (stripe fingerprint):**
1. If crop is dark (mean brightness < 50): apply gamma correction + strong CLAHE first
2. Resize to 224×224, normalise with ImageNet mean/std
3. Pass through ResNet50 up to `layer4` (before the classification head)
4. Extract the 2048-dimensional feature vector = the tiger's "stripe fingerprint"
5. Apply Global Average Pooling to get a 1D vector of 2048 numbers

**Step 8.2 — Identity matching:**
1. Load all existing tiger profiles from JSON database
2. For each profile: compute **cosine similarity** between new fingerprint and stored embedding
   - `similarity = (A · B) / (|A| × |B|)` — scale-invariant, handles brightness variation
3. Find the tiger with the highest cosine similarity score
4. If `best_score >= 0.83` → **MATCH** — this is a known tiger, update its profile
5. If `best_score < 0.83` → **NEW** — enrol as next Tiger_ID (Tiger_001, Tiger_002, ...)

**Step 8.3 — Running average embedding (DB update):**
- For matched tigers: `new_embedding = (old_embedding × n + new_fingerprint) / (n + 1)`
- Partial body / corner trace detections: match only, don't update DB (low-quality crop)
- Full body detections: match AND update DB — each sighting improves the fingerprint

> **Why cosine similarity (not Euclidean distance)?**
> Cosine similarity measures the *angle* between vectors, not their magnitude.
> A dark image and a bright image of the same tiger have very different magnitudes
> but nearly the same angle. Cosine is invariant to brightness scale changes.

---

## Stage 9 — Grad-CAM Explainability (optional, `--gradcam` flag)

```
Purpose: Prove the model is reading stripe patterns, not background
```

1. Register a forward hook on ResNet50 `layer4` to capture feature maps
2. Register a backward hook to capture gradients flowing back from the "tiger" class score
3. Run a forward pass on the crop
4. Back-propagate the gradient of the tiger class score (not the loss)
5. Global average pool the gradients over the spatial dimensions → weight per feature map
6. Weight each feature map by its gradient weight, sum, apply ReLU
7. Resize the resulting activation map to match the original crop size
8. Apply jet colourmap: red = highest activation (where model "looks"), blue = ignored
9. Overlay onto original image — produces the 3-panel heatmap

**How to read the heatmap:**
- Red/yellow on tiger body stripes → model is reading the right features ✓
- Red/yellow on background (trees, rocks) → model is using the wrong features ✗
- All Test Bh heatmaps show activation on body/stripes, not background → pipeline is valid

---

## Bug Fixes Applied (v3.0 → v3.0 Final)

| # | Bug | Root Cause | Fix |
|---|---|---|---|
| 1 | Corner Trace on Whole-Image Fallback | Fallback box (0,0,W,H) always touches border → wrongly labeled Corner_Trace | Added `if det_src != 'Whole_Image_Fallback'` check |
| 2 | IR images rejected as underexposed | Brightness check ran BEFORE IR detection | Moved IR check to execute FIRST in triage |
| 3 | Blur threshold too strict | 80 calibrated on DSLR images; camera traps score 38-48 | Lowered BLUR_THRESHOLD 80 → 25 |
| 4 | Shadow guard rejects IR images | IR = greyscale = saturation=0 = always triggers shadow guard | Added `if bucket != 'IR_Night'` condition |
| 5 | Species gate insufficient for unusual poses | Tiger lying flat → "hog/wild boar" in top-3; no retry | Added 3-attempt strategy (top-3 → brightened → top-5) |
| 6 | Leopard accepted via top-5 widening | top-5 found "tiger cat" (S. American wildcat) after leopard failed top-3 | Added leopard dominance rule: ≥2 non-tiger big cats in top-3 → immediate reject |
| 7 | COCO zebra class ID wrong | `TIGER_PROXY_CLASSES` had class 24 (backpack), not 22 (zebra) | Fixed to `{15, 16, 17, 22}` |
| 8 | OIV7 misses tigers after preprocessing | CLAHE + DCP suppresses YOLO feature maps | Dual scan: run OIV7/COCO on raw + preprocessed, merge via IoU deduplication |
| 9a | Multi-tiger frames counted as 1 | Whole-image fallback = 1 detection regardless of tigers in frame | Grid scan: split into halves, check each independently, enrol separately if both confirm |
| 9b | Single tiger counted as 2 (L/R split) | Head in left half + hindquarters in right half both pass species gate | Fingerprint similarity guard: if sim ≥ 0.68 → same tiger, keep 1 detection |
| 9c | Single tiger still counted as 2 (T/B split) | After L/R guard fires, code fell through to top/bottom split (sim=0.44 < 0.68 → 2 detections) | Added `grid_handled=True` flag when L/R reaches any conclusion; top/bottom gated behind `if not grid_handled` |

---

## Algorithms Used — Summary Table

| Algorithm | Stage | What It Does | Why This One |
|---|---|---|---|
| **YOLOv8n + OIV7 weights** | Detection Stage 1 | Draws bounding boxes — direct Tiger class | Only model with an actual "Tiger" class |
| **YOLOv8n + COCO weights** | Detection Stage 2 | Proxy detection via cat/dog/horse/zebra | Fallback when OIV7 misses (partial occlusion) |
| **CLAHE** | Preprocessing | Contrast enhancement (local, not global) | Recovers stripe texture in shadow/low-light without overexposing |
| **Dark Channel Prior** | Preprocessing | Haze/fog removal | Sundarbans monsoon fog washes out stripe detail |
| **Gamma correction** | Preprocessing | Lifts dark images to target brightness | Enables ResNet to see stripes in under-lit images |
| **ResNet50 (classifier)** | Species gate | Top-k label prediction | Pretrained on ImageNet which includes tiger; fast, proven transfer learning |
| **ResNet50 (feature extractor)** | Fingerprinting | 2048-dim stripe fingerprint from layer4 | layer4 features encode texture/stripe patterns, not just category |
| **Cosine similarity** | Identity matching | Compares fingerprints across images | Scale-invariant; handles brightness/sensor variation across camera traps |
| **Running average embedding** | DB update | Fingerprint improves with each sighting | Single image noisy; average of all sightings is more robust |
| **Grad-CAM** | Explainability | Shows which pixels the model activated on | Visual proof model reads stripes not background; required for viva |
| **IoU deduplication** | Dual scan merge | Prevents double-counting the same tiger box | IoU threshold 0.30 removes overlapping boxes from raw+preprocessed scans |
| **Grid scan** | Multi-tiger detection | Splits frame, checks halves independently | Detects mother+cub or side-by-side tigers when YOLO finds no individual boxes |

---

## Data Flow — One Image End to End

```
Camera Trap Image (JPG/PNG)
        │
        ▼
[STAGE 1] Load image → img (raw), img_proc (will be preprocessed)
        │
        ▼
[STAGE 2] Triage
    ├── IR detected? → IR_Night bucket (skip brightness/blur)
    ├── Too bright?  → SKIP (overexposed)
    ├── Too dark?    → SKIP (underexposed, threshold=8)
    └── Too blurry?  → SKIP (Laplacian<25)
        │
        ▼
[STAGE 3] Preprocess
    ├── IR path: greyscale → CLAHE → back to BGR
    └── Clean path: gamma correction → CLAHE → Dark Channel Prior dehazing
        │
        ▼
[STAGE 4] Cascade Detection
    ├── OIV7 on img_proc → Tiger boxes?  ──────────────────┐
    ├── OIV7 on img_raw  → more Tiger boxes? (dedup IoU)   │
    ├── COCO on img_proc → proxy boxes? (class 15/16/17/22)│
    ├── COCO on img_raw  → more proxy boxes?               │
    └── No boxes found   → Whole_Image_Fallback box        │
        │                                                   │
        ▼                                                   │
[STAGE 5] Species Gate (per box)                           │
    ├── OIV7 boxes ─────────────────────────── SKIP gate ──┘
    ├── Attempt 1: top-3 labels → tiger? → CONFIRM
    ├── Leopard dominance? (≥2 big cats) → REJECT immediately
    ├── Attempt 2: brightened crop → tiger in top-3? → CONFIRM
    └── Attempt 3: top-5 → tiger? → CONFIRM | else REJECT
        │
        ▼
[STAGE 6] Grid Scan (Whole_Image_Fallback only)
    ├── Split left/right → both confirm tiger? → 2 detections
    ├── Split top/bottom → both confirm tiger? → 2 detections
    └── One side fails → keep 1 detection
        │
        ▼
[STAGE 7] ViewPoint Classification
    └── box centre position → Left_Flank / Right_Flank / Frontal
        │
        ▼
[STAGE 8] Fingerprinting + Identity Match
    ├── Gamma + CLAHE on dark crop
    ├── ResNet50 layer4 → 2048-dim fingerprint
    ├── Cosine similarity vs all DB profiles
    ├── Score ≥ 0.83 → MATCHED (known tiger, update DB)
    └── Score < 0.83 → NEW (enrol as Tiger_XXX)
        │
        ▼
[STAGE 9] Grad-CAM (--gradcam flag)
    └── Gradient × feature map → heatmap overlay → saved to reports/ultimate_gradcam/
        │
        ▼
OUTPUTS
    ├── Annotated image (bounding boxes + Tiger_ID labels)
    ├── Population report CSV (one row per detection)
    ├── Tiger identity DB JSON (persistent across runs)
    └── Grad-CAM heatmap images
```

---

## Census Logic

After all images in a batch are processed:

1. Count unique Tiger_IDs from the DB
2. Group by ViewPoint: how many unique IDs seen from Left_Flank / Right_Flank / Frontal
3. **Conservative estimate** = `max(left_count, right_count, frontal_count)`
   - Prevents counting the same tiger twice (once left flank, once right flank)
4. **Liberal estimate** = union of all IDs seen from any viewpoint
   - Upper bound; assumes all detections are unique individuals

---

*Last updated: 2026-06-02 | EPAIB Batch 05, Group 4, IIM Lucknow*
*All 11 bug fixes (including grid scan over-counting fixes 9a/9b/9c) are live in `src/tiger_ai_ultimate.py`*
