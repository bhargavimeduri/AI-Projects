# 🐯 Sundarban Tiger Image Processing
### EPAIB Batch 05 — Group 4 | IIM Lucknow

---

## Project Overview

An AI-powered image processing system to **identify, classify, and analyze Bengal Tigers** in the Sundarbans using Computer Vision and Deep Learning.

The Sundarbans is home to the world's largest population of Bengal Tigers. This project applies modern image classification and object detection techniques to support wildlife conservation efforts through automated tiger identification from camera trap images.

---

## Problem Statement

Through our group member **Mr. Anamitra Lahiri**, we have established a direct connection with the **Programme Director, Wildlife Department, Government of West Bengal**. The department is actively exploring AI/ML to improve tiger enumeration in the Sundarbans Tiger Reserve.

Currently, the forest department relies on **manually analysing thousands of camera trap images** to identify individual tigers by stripe patterns. This is:
- **Time-intensive** — thousands of images reviewed by hand
- **Labour-intensive** — significant ranger effort diverted from field work
- **Error-prone** — fatigue and human bias affect accuracy

**Our Solution:** A Computer Vision-based system that:
1. **Detects** tigers automatically from camera trap images
2. **Identifies** individual tigers using stripe pattern recognition
3. **Estimates** population accurately and efficiently
4. **Scales** nationally across all tiger reserves in India

---

## Team — Group 4

| Name | Role | Contact |
|------|------|---------|
| Bhargavi Meduri | Project Lead / ML Engineer | bhargavimeduri |
| [Member 2] | Data Engineer | |
| [Member 3] | Model Development | |
| [Member 4] | Evaluation & Reporting | |
| [Member 5] | Presentation & Documentation | |

> Update this table with your team members.

---

## Repository Structure

```
epaib-batch05-group4/
│
├── README.md                   ← You are here
├── requirements.txt            ← All Python dependencies
├── .gitignore                  ← Files excluded from git
│
├── docs/                       ← All project documentation
│   ├── problem_statement.md    ← Detailed problem definition
│   ├── architecture.md         ← Model & system architecture
│   ├── dataset_guide.md        ← Dataset sources & structure
│   └── meeting_notes.md        ← Team meeting logs
│
├── notebooks/                  ← Jupyter notebooks (run in order)
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_evaluation.ipynb
│
├── src/                        ← Reusable Python modules
│   ├── preprocessing.py        ← Image preprocessing functions
│   ├── model.py                ← Model architecture definition
│   ├── train.py                ← Training pipeline
│   └── utils.py                ← Helper functions
│
├── data/
│   ├── raw/                    ← Original images (gitignored)
│   ├── processed/              ← Cleaned & resized images (gitignored)
│   └── sample/                 ← Small sample for testing (committed)
│
├── models/                     ← Saved model weights (gitignored)
├── reports/
│   ├── figures/                ← Charts, confusion matrix, sample outputs
│   └── final_report.md         ← IIM submission report
│
└── presentation/               ← Final slides for IIM presentation
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| Deep Learning | TensorFlow / PyTorch |
| Image Processing | OpenCV, PIL |
| Model Architecture | CNN / ResNet50 / EfficientNet |
| Object Detection | YOLO v8 |
| Experiment Tracking | MLflow / Weights & Biases |
| Notebooks | Jupyter |
| Version Control | Git + GitHub |

---

## Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/bhargavimeduri/epaib-batch05-group4.git
cd epaib-batch05-group4

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run notebooks in order
jupyter notebook notebooks/01_data_exploration.ipynb
```

---

## Data & Feasibility

| Item | Detail |
|------|--------|
| **Source** | Wildlife Department, Government of West Bengal |
| **Volume** | ~1 TB of historical camera trap images (with potential for more) |
| **Regulatory** | Permissions obtainable from the Government |
| **Constraints** | No masking or anonymisation required |
| **Connection** | Via group member Mr. Anamitra Lahiri (direct govt. contact) |
| **Deployment** | Real-world deployment potential — scalable across India |

> Raw data is NOT committed to this repo due to size. See `docs/dataset_guide.md` for structure and access details.

---

## Project Timeline

| Phase | Task | Status |
|-------|------|--------|
| Week 1 | Problem definition, dataset sourcing | 🔄 In Progress |
| Week 2 | Data exploration & preprocessing | ⏳ Pending |
| Week 3 | Model training (baseline CNN) | ⏳ Pending |
| Week 4 | Model training (transfer learning) | ⏳ Pending |
| Week 5 | Evaluation & optimisation | ⏳ Pending |
| Week 6 | Report writing & presentation | ⏳ Pending |

---

## Programme Details

- **Programme:** Executive Programme in AI for Business (EPAIB)
- **Institution:** IIM Lucknow
- **Batch:** 05
- **Group:** 4
- **Academic Year:** 2025–2026

---

## Contributing

1. Create your branch: `git checkout -b feature/your-name-task`
2. Make changes and commit: `git commit -m "Add: brief description"`
3. Push: `git push origin feature/your-name-task`
4. Create a Pull Request — tag Bhargavi for review

---

*Built with purpose — for the tigers of Sundarbans and the future of AI-driven conservation.* 🐯
