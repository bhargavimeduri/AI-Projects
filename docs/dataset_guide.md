# Dataset Guide
## Where to get data, how to structure it

---

## Recommended Datasets

### 1. Kaggle — Tiger Detection (Primary)
```
URL:    https://www.kaggle.com/datasets/
Search: "tiger detection" / "bengal tiger" / "wildlife camera trap"
Format: JPEG images with labels
Action: Download → place in data/raw/
```

### 2. iNaturalist — Open Wildlife Images
```
URL:    https://www.inaturalist.org/observations?taxon_id=42016
Filter: Panthera tigris tigris (Bengal Tiger)
Format: Open license images
```

### 3. Wildlife Insights (Google + WWF)
```
URL:    https://www.wildlifeinsights.org
Format: Camera trap images with AI-generated labels
Best for: Large volume, pre-labelled
```

### 4. LILA BC — Camera Trap Dataset
```
URL:    https://lila.science/datasets/
Search: Tiger / Big cats
Best for: Research-grade labelled camera trap images
```

---

## Expected Folder Structure (after download)

```
data/
├── raw/
│   ├── tiger/          ← images containing tigers
│   └── no_tiger/       ← images without tigers (background, other animals)
│
├── processed/
│   ├── train/
│   │   ├── tiger/
│   │   └── no_tiger/
│   ├── val/
│   │   ├── tiger/
│   │   └── no_tiger/
│   └── test/
│       ├── tiger/
│       └── no_tiger/
│
└── sample/             ← 20-30 images committed to repo for testing
    ├── tiger/
    └── no_tiger/
```

---

## Minimum Dataset Size

| Stage | Minimum Images | Recommended |
|-------|---------------|-------------|
| Detection (tiger/no tiger) | 500 per class | 2000+ per class |
| Individual ID | 50 per tiger | 200+ per tiger |
| YOLOv8 detection | 500 annotated | 1500+ annotated |

---

## Data Collection Notes

- Camera trap images are often **dark, blurry, or partially obscured** — this is expected
- Include **night vision / infrared** images — tigers are nocturnal
- Balance dataset — equal tiger and non-tiger images
- For individual ID — need multiple images of the **same tiger** from different angles

---

*Last updated: 2026-05-19 | Group 4 | EPAIB Batch 05 | IIM Lucknow*
