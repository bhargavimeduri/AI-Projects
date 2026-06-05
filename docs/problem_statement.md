# Problem Statement
## Sundarban Tiger Image Processing — EPAIB Batch 05, Group 4

---

## Background

The Sundarbans, spanning India and Bangladesh, is the world's largest mangrove forest and the primary habitat of the **Bengal Tiger (Panthera tigris tigris)**. As of 2023, approximately **500+ tigers** inhabit this region, making it one of the most critical tiger conservation zones globally.

Wildlife authorities rely on **camera traps** — motion-triggered cameras placed throughout the forest — to monitor tiger populations. However, this generates thousands of images that require **manual review**, creating a significant bottleneck in conservation monitoring.

### How This Project Came to Be

Through our group member **Mr. Anamitra Lahiri**, Group 4 has established a direct connection with the **Programme Director, Wildlife Department, Government of West Bengal**. The department is actively exploring AI/ML to modernise their tiger enumeration processes and has expressed strong interest in this project.

The department possesses approximately **~1 TB of historical camera trap image data** (with potential for more). Regulatory permissions are obtainable. No data masking or anonymisation constraints apply.

**Faculty Supervisor:** Dr. Sowmya S, Associate Professor, IIM Lucknow (sowmya@iiml.ac.in)

---

## The Problem

| Challenge | Impact |
|-----------|--------|
| Volume of camera trap images | Rangers cannot review all images in time |
| Manual identification is slow | Population counts are delayed by weeks |
| Human fatigue causes errors | Tigers are missed or miscounted |
| Individual tiger ID is difficult | Movement patterns cannot be tracked |
| No real-time alerts | Poaching or distress detection is delayed |

---

## Our Objective

Build an AI-powered image processing pipeline that:

1. **Detects** — Is a tiger present in this image? (Binary classification)
2. **Identifies** — Which individual tiger is this? (Multi-class classification using stripe patterns)
3. **Counts** — How many tigers appear in this image? (Object detection)
4. **Enumerates** — How many *unique* individual tigers exist across all images? (Identity tracking using ResNet50 embeddings + cosine similarity — same tiger appearing in 10 images counts as 1, not 10)
5. **Classifies Tiger Type** — What color morph is this tiger? (HSV stripe pattern analysis)
   - Orange Standard (most common Bengal tiger)
   - White Tiger (rare recessive gene)
   - Golden Tiger (tabby gene — extremely rare)
   - Black Tiger (pseudo-melanistic — very rare)
   - Snow White Tiger (albino variant)
6. **Alerts** — Flag unusual activity (injured tiger, human presence, poaching indicators)

---

## Success Criteria

| Metric | Target |
|--------|--------|
| Detection Accuracy | > 90% |
| Individual ID Accuracy | > 80% |
| Unique Tiger Count Consistency | Same count across 2 independent runs (reproducibility) |
| Color Morph Classification Consistency | Same morph per image across 2 independent runs |
| False Negative Rate | < 5% (missing a tiger is worse than a false alarm) |
| Processing Speed | < 2 seconds per image |

---

## Scope

**In Scope:**
- Image classification (tiger / no tiger)
- Individual tiger identification from stripe patterns
- Object detection for multiple tigers in one frame
- Unique tiger enumeration across all images (de-duplicated count)
- Tiger type classification by color morph (Orange / White / Golden / Black / Snow White)
- Model evaluation and performance reporting
- Reproducibility validation (devil's advocacy — two independent runs)

**Out of Scope:**
- Real-time camera trap integration (future phase)
- Behavioural analysis (future phase)
- Mobile deployment (future phase)

---

## Why This Matters

> Every tiger not counted is a data point lost for conservation.
> Every poaching event not detected is a tiger at risk.
> AI does not get tired. AI does not miss an image.

This project directly contributes to **wildlife conservation at scale** — a real-world application of AI for social good.

---

## Why This Project Matters

> This initiative has strong real-world and commercial potential, with applications across wildlife conservation efforts in India. If successful, it could significantly enhance the accuracy and efficiency of tiger population monitoring on a national scale.

**Impact at scale:**
- Replace manual image review across Sundarbans (~1TB of images)
- Potential deployment to **all tiger reserves across India**
- Contribute to national tiger census accuracy
- Enable real-time monitoring and early alert systems

---

## Mentor Request (Deep Learning & CV Expert)

We are seeking mentorship from a Deep Learning and Computer Vision expert. Specifically we need guidance on:

| Area | What We Need |
|------|-------------|
| Problem structuring | Validate our approach and scope |
| Model selection | Architecture decisions — CNN vs transformer vs hybrid |
| Execution roadmap | Best practices, pitfalls to avoid |
| Periodic reviews | Feedback at key milestones |

---

*Last updated: 2026-05-23 | Group 4 | EPAIB Batch 05 | IIM Lucknow*
