# Problem Statement
## Sundarban Tiger Image Processing — EPAIB Batch 05, Group 4

---

## Background

The Sundarbans, spanning India and Bangladesh, is the world's largest mangrove forest and the primary habitat of the **Bengal Tiger (Panthera tigris tigris)**. As of 2023, approximately **500+ tigers** inhabit this region, making it one of the most critical tiger conservation zones globally.

Wildlife authorities rely on **camera traps** — motion-triggered cameras placed throughout the forest — to monitor tiger populations. However, this generates thousands of images daily that require **manual review**, creating a significant bottleneck in conservation monitoring.

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
4. **Alerts** — Flag unusual activity (injured tiger, human presence, poaching indicators)

---

## Success Criteria

| Metric | Target |
|--------|--------|
| Detection Accuracy | > 90% |
| Individual ID Accuracy | > 80% |
| False Negative Rate | < 5% (missing a tiger is worse than a false alarm) |
| Processing Speed | < 2 seconds per image |

---

## Scope

**In Scope:**
- Image classification (tiger / no tiger)
- Individual tiger identification from stripe patterns
- Object detection for multiple tigers in one frame
- Model evaluation and performance reporting

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

*Last updated: 2026-05-19 | Group 4 | EPAIB Batch 05 | IIM Lucknow*
