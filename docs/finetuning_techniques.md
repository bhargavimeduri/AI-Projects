# Fine-Tuning Techniques for Neural Networks
## Applied Context — TRACE Tiger Pipeline
**EPAIB Batch 05 | Group 4 | IIM Lucknow**
**Date:** 2026-06-05

---

## What is Fine-Tuning?

Fine-tuning is the process of taking a neural network that has already been trained on a large dataset and adapting it to perform better on your specific, smaller dataset. Instead of training from scratch (which requires millions of images), you start from a model that already understands shapes, textures, and patterns — and teach it the specific differences that matter for your task.

> **In TRACE:** Our ResNet50 already knows what a tiger looks like from ImageNet. Fine-tuning teaches it to distinguish *Tiger_001's stripes* from *Tiger_002's stripes*.

---

## Technique 1 — Transfer Learning

**What it is:** Use a model pre-trained on a large dataset (ImageNet — 14 million images, 1,000 classes) and adapt it to your specific task.

**How it works — Two phases:**

**Phase 1 — Frozen Base (Fast Learning)**
- All base layers are locked (frozen)
- Only the top classification layers are trained
- High learning rate (1e-4) — learns quickly
- Goal: teach the new head what tiger vs not-tiger means

**Phase 2 — Fine-Tuning (Precision)**
- Top N layers of the base are unfrozen
- Very low learning rate (1e-5) — small, careful updates
- Goal: adjust the deep features to tiger stripe patterns specifically

**Applied in TRACE:**
- ResNet50 → frozen base → trained top → unfrozen top 30 layers
- EfficientNetB3 → same two-phase approach for individual tiger ID

**Why it matters for us:**
We only have 187 images. Training ResNet50 from scratch needs millions. Transfer learning solves the small-dataset problem.

---

## Technique 2 — Learning Rate Scheduling

**What it is:** Control how fast the model updates its weights at each point in training. A fixed learning rate is rarely optimal — too high causes instability, too low causes slow convergence.

| Schedule | How it Works | Best For |
|---|---|---|
| **Warm-up** | Start very low (1e-6), increase gradually over first 5 epochs | Prevents early instability when unfreezing layers |
| **Cosine Annealing** | LR follows a cosine curve — smooth decay from max to min | General fine-tuning, prevents abrupt drops |
| **ReduceLROnPlateau** | Cut LR by 50% when val loss plateaus for N epochs | Adaptive — responds to actual training behaviour |
| **Cyclic LR** | Oscillate between min and max LR repeatedly | Helps escape local minima in complex loss landscapes |

**Applied in TRACE:** ReduceLROnPlateau (factor=0.5, patience=3) — already in our training code.

**Recommended addition:** Warm-up for the first 3 epochs when fine-tuning YOLOv8l, as large models are sensitive to initial updates.

---

## Technique 3 — Data Augmentation

**What it is:** Artificially expand your training dataset by creating modified versions of existing images. The model sees the same tiger in different conditions — making it robust to real-world variation.

| Technique | What it Does | Tiger Application |
|---|---|---|
| **Horizontal Flip** | Mirror the image | Same tiger seen from opposite side |
| **Brightness Jitter** | Random brighten/darken ±30% | Day vs dusk camera trap conditions |
| **Contrast Jitter** | Vary contrast | Overcast vs sunny forest lighting |
| **Rotation (±15°)** | Slight tilt | Camera trap angles vary |
| **Zoom / Crop** | Random zoom in/out | Tigers at different distances from camera |
| **Mosaic** | Combine 4 images into one training sample | **Directly improves multi-tiger detection** |
| **MixUp** | Blend two images at pixel level (α=0.5) | Robust to partial occlusion |
| **CutMix** | Cut a patch from image A, paste into image B | Handles tigers partially hidden by foliage |

**Most critical for TRACE:**
- **Mosaic augmentation** — directly teaches YOLO to detect multiple tigers in one frame. This is why 000082 and 000102 had missed second tigers — YOLO was never trained with multi-tiger mosaic samples.

---

## Technique 4 — Regularisation

**What it is:** Techniques that prevent the model from memorising training data (overfitting) and force it to learn general patterns. Critical when dataset is small.

| Technique | How it Works | Applied In TRACE |
|---|---|---|
| **Dropout** | Randomly deactivates neurons during training | Dropout(0.5) after Dense layers in detection model |
| **L2 Weight Decay** | Penalises large weights — keeps model general | Via Adam optimiser weight_decay parameter |
| **Label Smoothing** | Instead of hard 0/1 labels, use 0.05/0.95 — reduces overconfidence | Can be added to classification loss |
| **Early Stopping** | Stop training when val metric stops improving for N epochs | patience=5 already in our EarlyStopping callback |
| **Batch Normalisation** | Normalises activations between layers — stabilises training | Built into ResNet50 and EfficientNetB3 base |

**Most critical for TRACE:**
- Label Smoothing — our dataset has uncertain labels (camera trap images where tiger is partially visible). Soft labels better reflect this uncertainty.

---

## Technique 5 — Few-Shot Fine-Tuning

**What it is:** Techniques specifically designed for situations where you have very few images per class. In TRACE, we may have only 3-5 images of Tiger_001 — standard classification breaks down at this scale.

| Technique | What it Does | How it Applies to TRACE |
|---|---|---|
| **Prototype Networks** | Compute a single "prototype" embedding per tiger by averaging all its image embeddings | Tiger_001's prototype = mean of its 5 embedding vectors. New image matches nearest prototype. |
| **Siamese Networks** | Train on *pairs* of images — predict "same tiger?" or "different tiger?" | More training signal from fewer images — each pair is a training sample |
| **Contrastive Learning** | Push same-tiger embeddings close together, push different-tiger embeddings apart in vector space | Improves the quality of our 2048-dim fingerprint |
| **Triplet Loss** | Train on triplets: Anchor (Tiger A), Positive (Tiger A again), Negative (Tiger B). Minimise Anchor-Positive distance, maximise Anchor-Negative | More precise than cosine similarity alone |
| **MAML** | Meta-learning — train the model to *learn fast from few examples* | Overkill for current scale, but relevant for national deployment |

**Most critical for TRACE:**
- **Contrastive Learning + Triplet Loss** — our current cosine similarity matching (threshold 0.83) has false matches because ResNet50 was not trained with wildlife-specific contrastive objectives. MegaDescriptor uses triplet loss trained on 37 wildlife species — this is why it is recommended as an upgrade.

**Current TRACE approach vs Few-Shot:**
```
Current:   Image → ResNet50 (ImageNet) → 2048-dim vector → Cosine Similarity
Upgraded:  Image → MegaDescriptor (Triplet Loss, Wildlife) → 512-dim vector → Cosine Similarity
```
The upgraded vector is smaller but far more discriminative for wildlife stripe patterns.

---

## Technique 6 — Knowledge Distillation

**What it is:** Train a small, fast model (student) to replicate the behaviour of a large, accurate model (teacher). Get large-model accuracy at small-model speed.

**How it works:**
1. Train the large model (YOLOv8l-OIV7) — the teacher
2. Run the teacher on all training images → collect its soft probability outputs
3. Train the small model (YOLOv8n-OIV7) to match the teacher's outputs, not just the hard labels

**Why it matters for TRACE:**
- YOLOv8l runs well on GPU but is slow on CPU
- For forest department deployment on field laptops → distil YOLOv8l into YOLOv8n
- Get the accuracy of the large model at the speed of the small model

---

## Technique 7 — Hyperparameter Tuning

**What it is:** Systematically search for the best combination of training settings.

| Hyperparameter | What to Tune | TRACE Range |
|---|---|---|
| Learning rate | Initial LR | 1e-3 to 1e-5 |
| Batch size | Images per gradient update | 8, 16, 32 |
| Confidence threshold | YOLO detection cutoff | 0.05 to 0.35 |
| Similarity threshold | Re-ID match cutoff | 0.75 to 0.90 |
| Blur threshold | Laplacian cutoff | 10 to 30 |
| Dropout rate | Regularisation strength | 0.3 to 0.6 |

**Methods:**
- **Grid Search** — try all combinations (slow but thorough)
- **Random Search** — sample random combinations (faster, often as good)
- **Bayesian Optimisation** — learns from previous trials to pick next best settings (most efficient)

**Most critical for TRACE:**
- Similarity threshold (currently 0.83) — determines whether two sightings are the same tiger
- Confidence threshold (OIV7 = 0.09, COCO = 0.35) — determines what YOLO accepts as a detection

---

## Priority Matrix for TRACE

| Priority | Technique | Problem it Solves | Effort |
|---|---|---|---|
| 🔴 Critical | Mosaic Augmentation | Missed second tiger in 000082, 000102 | Low — add to training config |
| 🔴 Critical | Contrastive / Triplet Loss | Improve stripe fingerprint matching | Medium — replace ResNet50 re-ID layer |
| 🟡 High | Orientation-aware Grid Split | False split on vertical image 000084 | Low — 3 lines of code |
| 🟡 High | BRISQUE Blur Detection | 000143 wrongly rejected | Low — replace Laplacian |
| 🟡 High | Label Smoothing | Soft labels for uncertain camera trap images | Low — one parameter change |
| 🟢 Medium | Few-Shot Prototype Networks | Better ID with 3-5 images per tiger | Medium |
| 🟢 Medium | Knowledge Distillation | Deploy YOLOv8l accuracy at YOLOv8n speed | High — post-deployment |
| 🟢 Low | MAML Meta-Learning | National scale deployment | High — future work |

---

## Summary

Fine-tuning is not one technique — it is a collection of strategies applied at different stages:

```
DATA LEVEL        → Augmentation (Mosaic, MixUp, CutMix)
TRAINING LEVEL    → LR Scheduling, Regularisation, Early Stopping
MODEL LEVEL       → Transfer Learning, Contrastive Loss, Triplet Loss
ARCHITECTURE      → Few-Shot (Siamese, Prototype, MAML)
DEPLOYMENT        → Distillation, Quantisation, Pruning
```

For TRACE specifically, the highest-impact changes are at the **Data Level** (mosaic augmentation for multi-tiger) and **Model Level** (contrastive loss for better re-ID) — both achievable without changing the overall pipeline architecture.

---

*Document prepared for EPAIB Batch 05, Group 4, IIM Lucknow*
*TRACE Tiger Pipeline — v3.0*
