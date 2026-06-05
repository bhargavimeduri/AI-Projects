# Yeshvir's Code Review
## EPAIB Batch 05 | Group 4 | IIM Lucknow
### Files reviewed: Tiger_Finetune_Colab.txt | Tiger_Finetune_Multi_Animal.txt

---

## First — Why Neither Script Will Run As-Is

Both scripts start with:
```python
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
```

**TensorFlow is NOT installed on this machine.** Our entire TRACE pipeline runs on PyTorch
(a different AI framework also made by Meta/Facebook). TensorFlow is made by Google.
Both do the same job — they are competitors. We cannot mix them.

The scripts will crash on line 1 with: `ModuleNotFoundError: No module named 'tensorflow'`

**What we did:** We ported both scripts to PyTorch so they can actually run on our machine
using the same libraries as TRACE. The ported versions are in `src/yeshvir/`.

---

## What Each Script Does — Plain English

### Script A: Tiger_Finetune_Colab.txt (Simpler Version)

**The approach:**
1. Load an image
2. Check if it is blurry or flat (quality check)
3. Resize the WHOLE image to 224×224 and feed it to ResNet50
4. ResNet50 classifies the whole image — if the top prediction contains "tiger", keep it
5. Extract the 2048-number feature vector from the same image
6. Compare against known tigers using cosine similarity
7. Assign a tiger ID (TIG_2026_0001, TIG_2026_0002...)

**What is good about it:**
- Clean, readable code
- Correct use of cosine similarity for identity matching
- Proper feature extraction from the avg_pool layer (2048-dim)
- Good CSV output format with status (Original / Duplicate)

**What is wrong about it:**

| Problem | Plain English Explanation | Why It Matters |
|---|---|---|
| No YOLO — whole image fed to ResNet50 | The whole image (tiger + trees + background) is sent to ResNet50. ResNet50 sees the background MORE than the tiger and will classify it as "forest" or "grass" not "tiger". | Will reject most valid tiger images because the background dominates. |
| top-1 must say "tiger" | ResNet50's top prediction must literally be "tiger". In camera trap images, the top prediction is often "cheetah", "tabby", or "jaguar" because of lighting and angle — even for a real tiger. | High false rejection rate — many real tigers will be dropped. |
| Two forward passes | Calls `base_model.predict()` then `feature_model.predict()` separately on the same image. Double the compute for no benefit. | Wasteful — should be one pass. |
| No preprocessing | No fog removal, no contrast enhancement. Dark monsoon images will fail quality checks. | Bad performance on real camera trap images. |
| No DB persistence | Features stored in memory only. If the program stops, all tiger IDs are lost. | Cannot build a census across multiple sessions. |
| No running average | New sighting of same tiger does not update or improve its stored fingerprint. | DB never improves with more data. |
| No partial body protection | A tail crop or corner trace can be used to create a new tiger profile, corrupting identity. | Same tiger gets multiple IDs. |

---

### Script B: Tiger_Finetune_Multi_Animal.txt (More Complex Version)

**The approach:**
1. Use YOLO to detect all animal-like shapes in the image
2. If YOLO finds nothing → use the whole image as a fallback
3. For each detected crop: quality check
4. ResNet50 checks if "tiger" appears in top-3 predictions → if yes, extract features
5. Cosine similarity matching → assign ID

**What is good about it:**
- **Uses YOLO for detection** — much better than Script A. Gets a crop of just the tiger before feeding to ResNet50
- **Top-3 check** (not top-1) — more tolerant, "tiger" just needs to be anywhere in top 3 predictions
- **Single forward pass** — the unified model runs both classification and feature extraction simultaneously. Smart design.
- **Fallback logic** — if YOLO finds nothing, uses the full image. Prevents completely missing an image.
- **Handles fallback vs YOLO box formats correctly** — the is_fallback flag is a genuine edge case well-handled

**What is wrong about it:**

| Problem | Plain English Explanation | Why It Matters |
|---|---|---|
| BLUR_THRESHOLD = 5.0 | Almost impossibly low. Comment says "don't reject soft/distant crops." But at 5.0, even a completely blurred, unidentifiable smear passes. TRACE uses 80.0. | Blurry, unusable crops will be accepted and create bad fingerprints. |
| DETECTION_CONF = 0.15 | Same problem as the Colab script we reviewed — 85% uncertainty accepted. | Shadows, rocks, stumps get detected and processed. |
| Wrong COCO class set | Includes elephant (20), bear (21), giraffe (23). There are NO elephants, bears, or giraffes in Sundarbans. These extra classes only add false positives. | Unrelated animals get picked up as tiger candidates. |
| `break` bug in matching | Stops searching at the FIRST match above 0.88. Never checks if another tiger in the DB is a BETTER match. | Wrong tiger gets the match if two tigers have similar appearances. |
| No running average | Same as Script A — DB never improves. | Weaker identity over time. |
| No DB persistence | Memory only — restart = start from zero. | Cannot build cumulative census. |
| No preprocessing | No CLAHE, no Dark Channel Prior fog removal. | Poor performance on monsoon/dark images. |
| yolov8m.pt | Uses the medium YOLO model (not nano). 5x slower on CPU. | On a basic laptop, processing 187 images will take 30+ minutes. |
| No partial body guard | Any crop updates the DB, even a partial body. | Same tiger gets multiple IDs. |
| New tiger always "100%" | A new tiger's Match_Confidence_Score is hardcoded to "100.0%" | Misleading. Should say "No match found" or "N/A". |

---

## The `break` Bug — Explained in Detail

This is in both scripts. Here is the code:

```python
for existing_id, existing_vector in valid_tiger_features:
    similarity = cosine_similarity(feat_vector, existing_vector)[0][0]
    if similarity > max_similarity:
        max_similarity = similarity
        if similarity >= SIMILARITY_THRESHOLD:
            matched_id = existing_id
            break   # ← BUG HERE
```

**What it does:** As soon as it finds ONE tiger with similarity >= 0.88, it stops looking.

**What it should do:** Check ALL known tigers and find the BEST match.

**Why this matters:** Imagine you have Tiger_001 with similarity 0.89 and Tiger_002
with similarity 0.96. The current code finds Tiger_001 first (0.89 ≥ 0.88), sets
matched_id = Tiger_001, and stops. Tiger_002 is never checked.
The image belongs to Tiger_002 but gets assigned to Tiger_001.

**The fix:**
```python
# Don't break early — find the BEST match across ALL known tigers
for existing_id, existing_vector in valid_tiger_features:
    similarity = cosine_similarity(feat_vector, existing_vector)[0][0]
    if similarity > max_similarity:
        max_similarity = similarity
        matched_id = existing_id if similarity >= SIMILARITY_THRESHOLD else None
# No break — let the loop complete
```

---

## Side-by-Side Comparison: Yeshvir's Scripts vs TRACE

| Feature | Script A (Colab) | Script B (Multi-Animal) | TRACE (Our Pipeline) |
|---|---|---|---|
| YOLO detection | No — whole image | Yes — yolov8m | Yes — yolov8n |
| Preprocessing (fog/haze) | No | No | CLAHE + Dark Channel Prior |
| Tiger verification method | ResNet50 top-1 = "tiger" | ResNet50 top-3 contains "tiger" | YOLO COCO proxy classes |
| Feature extraction | 2048-dim avg_pool | 2048-dim avg_pool (same) | 2048-dim avg_pool (same) |
| Cosine similarity | Yes | Yes | Yes |
| Similarity threshold | 0.88 | 0.88 | 0.83 |
| Best match vs first match | First match (bug) | First match (bug) | Best match (correct) |
| Running average embedding | No | No | Yes |
| DB persistence | No | No | JSON file |
| Partial body guard | No | No | Yes |
| Shadow detection | No | No | Yes |
| Corner trace detection | No | No | Yes |
| Grad-CAM | No | No | Yes |
| Framework | TensorFlow (not installed) | TensorFlow (not installed) | PyTorch (installed) |
| Can run on this machine | No | No | Yes |

---

## What Is Genuinely Good in Yeshvir's Work

1. **The fallback logic in Script B** — if YOLO finds nothing, use the whole image.
   This is a practical real-world safeguard that TRACE does not have.

2. **Single forward pass (unified model)** in Script B — running classification
   and feature extraction in one pass is efficient design. We do two passes in TRACE.

3. **The expanded animal class set** idea is interesting — though the specific
   classes chosen are wrong for Sundarbans, the concept of being more permissive
   about what YOLO accepts (then filtering with ResNet50) is valid.

4. **TIG_2026_XXXX ID format** — a more human-readable ID format than Tiger_001.
   The year prefix makes records easier to track across survey years.

5. **Both scripts are clean and readable** — well-commented, good stage labels
   (STAGE 1, STAGE 2...), easy to follow for a non-technical reviewer.

---

## Executable Versions

Because TensorFlow is not installed, we created PyTorch ports of both scripts.
They produce the same output but use PyTorch (which TRACE already uses).

| Original | Ported Version | Location |
|---|---|---|
| Tiger_Finetune_Colab.txt | yeshvir_colab_pytorch.py | src/yeshvir/ |
| Tiger_Finetune_Multi_Animal.txt | yeshvir_multi_animal_pytorch.py | src/yeshvir/ |

Key fixes applied in the ports:
- TensorFlow → PyTorch
- `break` bug fixed — finds best match, not first match
- BLUR_THRESHOLD corrected from 5.0 to 50.0 (Script B)
- DETECTION_CONF corrected from 0.15 to 0.30 (Script B)
- COCO class set corrected to {15,16,17,24} (Script B)
- IMAGE_FOLDER pointed to our actual dataset
- New tiger confidence changed from "100.0%" to "N/A (New Enrollment)"

---

*Reviewed by Group 4 | EPAIB Batch 05 | IIM Lucknow | 2026-06-02*
