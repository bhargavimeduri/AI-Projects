# System Architecture
## Sundarban Tiger Image Processing Pipeline

---

## High-Level Architecture

```
CAMERA TRAP IMAGE
      │
      ▼
┌─────────────────────┐
│   DATA PIPELINE     │
│  - Image ingestion  │
│  - Resize & normalise│
│  - Augmentation     │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   STAGE 1           │
│  DETECTION MODEL    │  ← Binary CNN: Tiger / No Tiger
│  (ResNet50)         │
└────────┬────────────┘
         │ Tiger detected
         ▼
┌─────────────────────┐
│   STAGE 2           │
│  IDENTIFICATION     │  ← Multi-class: Which tiger?
│  (EfficientNet)     │     Uses stripe pattern analysis
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   STAGE 3           │
│  OBJECT DETECTION   │  ← YOLOv8: Count tigers in frame
│  (YOLOv8)           │     Draw bounding boxes
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   OUTPUT            │
│  - Tiger ID         │
│  - Count            │
│  - Confidence score │
│  - Alert flag       │
└─────────────────────┘
```

---

## Model Architecture Details

### Stage 1 — Detection (ResNet50 Transfer Learning)
```
Input:  224×224×3 image
Base:   ResNet50 (pretrained on ImageNet — frozen layers)
Head:   GlobalAveragePooling → Dense(256, relu) → Dropout(0.5) → Dense(1, sigmoid)
Output: Probability [0-1] → Tiger(1) / No Tiger(0)
Loss:   Binary Cross-Entropy
```

### Stage 2 — Individual ID (EfficientNetB3)
```
Input:  300×300×3 cropped tiger image
Base:   EfficientNetB3 (pretrained — fine-tuned last 20 layers)
Head:   GlobalAveragePooling → Dense(512, relu) → Dense(N_tigers, softmax)
Output: Tiger ID class probability
Loss:   Categorical Cross-Entropy
```

### Stage 3 — Detection & Count (YOLOv8)
```
Input:  640×640 image
Model:  YOLOv8m (medium) — fine-tuned on tiger dataset
Output: Bounding boxes + confidence scores + count
```

---

## Data Flow

```
Raw Images (camera trap)
    │
    ├── Train set (70%)  ─────────────────► Model Training
    ├── Validation set (15%)  ───────────► Hyperparameter Tuning
    └── Test set (15%)  ──────────────────► Final Evaluation
```

---

## Technology Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Base model | ResNet50 + EfficientNet | Proven on image classification, good transfer learning |
| Object detection | YOLOv8 | Fast, accurate, great for wildlife images |
| Framework | TensorFlow/Keras | Team familiarity, good documentation |
| Experiment tracking | MLflow | Open source, easy to use |
| Image augmentation | Albumentations | Rich augmentation library for wildlife images |

---

*Last updated: 2026-05-19 | Group 4 | EPAIB Batch 05 | IIM Lucknow*
