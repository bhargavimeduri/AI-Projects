<div align="center">

# 🐯 AI-Driven Tiger Enumeration
## Using Computer Vision

![Tiger](https://img.shields.io/badge/Project-Tiger%20Enumeration-orange?style=for-the-badge&logo=tensorflow)
![IIM](https://img.shields.io/badge/IIM%20Lucknow-EPAIB%20Batch%2005-darkblue?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)
![Data](https://img.shields.io/badge/Data-~1TB%20Camera%20Trap-yellow?style=for-the-badge)
![Gov](https://img.shields.io/badge/Partner-Govt%20of%20West%20Bengal-green?style=for-the-badge)

> **"From thousands of unreviewed images to a real-time national tiger census — powered by AI."**

**Group 4 | EPAIB Batch 05 | IIM Lucknow | Faculty: Dr. Sowmya S**

</div>

---

## 🌿 The Problem

The **Sundarbans Tiger Reserve** — world's largest mangrove forest, home to 500+ Bengal Tigers.

The forest department collects **thousands of camera trap images every week**. Every single one is reviewed manually by rangers to identify individual tigers by their stripe patterns.

```
❌  Time-intensive      →  Weeks to process one batch
❌  Labour-intensive    →  Rangers diverted from fieldwork
❌  Error-prone         →  Human fatigue misses tigers
❌  Not scalable        →  Cannot cover all reserves nationally
```

---

## ✅ Our Solution

A **3-stage AI pipeline** that makes tiger enumeration automatic, accurate, and scalable.

```
📸 Camera Trap Image
        │
        ▼
┌───────────────────┐
│  STAGE 1          │  🔍 ResNet50 — Is there a tiger?
│  DETECTION        │     Tiger / No Tiger  (target: >90% accuracy)
└────────┬──────────┘
         │ ✅ Tiger found
         ▼
┌───────────────────┐
│  STAGE 2          │  🐾 EfficientNetB3 — WHICH tiger?
│  IDENTIFICATION   │     Stripe pattern recognition
└────────┬──────────┘     Individual ID from database
         │
         ▼
┌───────────────────┐
│  STAGE 3          │  🔢 YOLOv8 — HOW MANY tigers?
│  COUNTING         │     Bounding boxes + count per frame
└────────┬──────────┘
         │
         ▼
📊 OUTPUT:  Count=2  |  T-17 ✓  |  T-23 ✓  |  2:14AM  |  Sector B
```

---

## 🏛️ Real-World Partnership

> 🤝 **Direct government connection established** via group member Mr. Anamitra Lahiri with the Programme Director, Wildlife Department, Government of West Bengal.

| Detail | Info |
|--------|------|
| 🗃️ **Data available** | ~1 TB historical camera trap images |
| 📜 **Regulatory** | Permissions obtainable from Government |
| 🔓 **Constraints** | No masking or anonymisation required |
| 🚀 **Deployment goal** | Scalable to all tiger reserves across India |

---

## 👥 Team — Group 4

| Name | Role |
|------|------|
| **Bhargavi Meduri** | Project Lead / ML Engineer |
| **Anamitra Lahiri** | Government Liaison / Domain Expert |
| [Member 3] | Data Engineering |
| [Member 4] | Model Development |
| [Member 5] | Evaluation & Reporting |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| 🧠 Detection | ResNet50 (Transfer Learning) |
| 🐾 Individual ID | EfficientNetB3 (Fine-tuned) |
| 🔢 Counting | YOLOv8 (Object Detection) |
| 📊 Similarity | Cosine Embeddings (Open-set ID) |
| 🔁 Augmentation | Albumentations |
| 📈 Tracking | MLflow |
| 🐍 Language | Python 3.10+ / TensorFlow |

---

## 📁 Repository Structure

```
epaib-batch05-group4/
│
├── 📓 notebooks/
│   ├── 01_data_exploration.ipynb         ← Understand your data first
│   ├── 02_preprocessing_augmentation.ipynb  ← Build the data pipeline
│   ├── 03_model_training.ipynb           ← Train + Grad-CAM visualisation
│   └── 04_evaluation_insights.ipynb      ← Metrics + conservation impact
│
├── 🐍 src/
│   ├── model.py          ← ResNet50 + EfficientNetB3 + Embeddings
│   ├── preprocessing.py  ← CLAHE + Augmentation pipeline
│   ├── train.py          ← Production training script (MLflow)
│   ├── yolo_detector.py  ← YOLOv8 tiger counter
│   └── utils.py          ← Plots, confusion matrix, metrics
│
├── 📚 docs/
│   ├── problem_statement.md  ← Full problem + govt partnership
│   ├── architecture.md       ← 3-stage pipeline design
│   ├── dataset_guide.md      ← Data sources + structure
│   └── meeting_notes.md      ← Team meeting logs
│
├── 🗄️ data/
│   ├── raw/        ← Original images (gitignored — 1TB)
│   ├── processed/  ← Preprocessed (gitignored)
│   └── sample/     ← Small sample committed for testing
│
└── 📊 reports/figures/   ← Confusion matrix, Grad-CAM, t-SNE plots
```

---

## 🚀 Getting Started

```bash
# Clone
git clone https://github.com/bhargavimeduri/epaib-batch05-group4.git
cd epaib-batch05-group4

# Setup
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Run notebooks in order
jupyter notebook
```

---

## 📅 Project Timeline

| Phase | Task | Status |
|-------|------|--------|
| 🔵 Week 1 | Problem definition + govt data request | ✅ Done |
| 🟡 Week 2 | Data receipt + exploration (Notebook 01) | ⏳ Awaiting data |
| 🟡 Week 3 | Preprocessing + augmentation pipeline | ⏳ Pending |
| 🟡 Week 4 | Detection model — ResNet50 + Grad-CAM | ⏳ Pending |
| 🟡 Week 5 | YOLOv8 counting + Individual ID | ⏳ Pending |
| 🟡 Week 6 | Evaluation + IIM presentation | ⏳ Pending |

---

## 🎯 Success Metrics

| Metric | Target |
|--------|--------|
| Tiger detection accuracy | > 90% |
| Individual ID accuracy | > 80% |
| False Negative Rate | < 5% |
| Processing speed | < 2 sec / image |

---

## 🏫 Programme Details

| | |
|---|---|
| **Programme** | Executive Programme in AI for Business (EPAIB) |
| **Institution** | IIM Lucknow |
| **Batch** | 05 |
| **Group** | 4 |
| **Faculty Supervisor** | Dr. Sowmya S — sowmya@iiml.ac.in |

---

<div align="center">

*Built with purpose — for the tigers of Sundarbans and the future of AI-driven conservation.* 🐯🌿

**If this works in Sundarbans, it works everywhere.**

</div>
