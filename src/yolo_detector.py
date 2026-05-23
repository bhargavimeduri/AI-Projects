"""
yolo_detector.py
Stage 3 — YOLOv8 Tiger Counter.
Detects and counts tigers per frame using bounding boxes.

Usage:
    detector = TigerDetector(model_path="models/yolo_tiger.pt")
    count = detector.count_tigers("data/raw/frame_001.jpg")
    annotated = detector.annotate_image("data/raw/frame_001.jpg")
    results_df = detector.batch_detect("data/raw/")
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


@dataclass
class DetectionResult:
    """One detected tiger in a frame."""
    bbox: tuple[int, int, int, int]   # (x1, y1, x2, y2) in pixels
    confidence: float                  # 0–1
    crop: np.ndarray                   # cropped tiger region (for ID stage)


class TigerDetector:
    """
    YOLOv8-based tiger detector.

    Stage 3 of the pipeline:
        ResNet50 (detect?) → EfficientNetB3 (who?) → YOLOv8 (how many?)

    The YOLOv8 model handles multi-tiger frames — bounding box per individual.
    Each crop can then be passed to the identification model.

    Args:
        model_path: Path to trained .pt weights file.
            Use "yolov8n.pt" (nano) for fast inference,
                "yolov8m.pt" (medium) for better accuracy.
        confidence_threshold: Minimum confidence to count a detection (default 0.5).
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
    ):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics not installed. Run: pip install ultralytics"
            )
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

    def detect(self, image_path: str) -> list[DetectionResult]:
        """
        Run detection on a single image.
        Returns list of DetectionResult — one entry per tiger found.

        Example:
            results = detector.detect("frame_001.jpg")
            for r in results:
                print(f"Tiger at {r.bbox}, confidence={r.confidence:.2f}")
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot load: {image_path}")

        results = self.model(image_path, conf=self.confidence_threshold, verbose=False)
        detections = []

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            crop = img[y1:y2, x1:x2]
            detections.append(DetectionResult(
                bbox=(x1, y1, x2, y2),
                confidence=conf,
                crop=crop,
            ))

        return detections

    def count_tigers(self, image_path: str) -> int:
        """
        Return just the count of tigers in one image.
        Convenience wrapper around detect().
        """
        return len(self.detect(image_path))

    def annotate_image(self, image_path: str) -> np.ndarray:
        """
        Draw bounding boxes and confidence scores on the image.
        Returns annotated BGR numpy array (can be saved with cv2.imwrite).

        Example:
            annotated = detector.annotate_image("frame_001.jpg")
            cv2.imwrite("output/annotated_frame_001.jpg", annotated)
        """
        img = cv2.imread(image_path)
        detections = self.detect(image_path)

        for i, det in enumerate(detections, 1):
            x1, y1, x2, y2 = det.bbox
            label = f"Tiger {i} ({det.confidence:.0%})"

            # Bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 165, 255), thickness=2)

            # Label background
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x1, y1 - h - 8), (x1 + w, y1), (0, 165, 255), -1)

            # Label text
            cv2.putText(img, label, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Count overlay in top-left
        count_text = f"Tigers: {len(detections)}"
        cv2.putText(img, count_text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        return img

    def batch_detect(self, image_dir: str) -> pd.DataFrame:
        """
        Run detection on all images in a directory.
        Returns DataFrame with columns:
            image_path, tiger_count, max_confidence, detections_json

        Example:
            df = detector.batch_detect("data/raw/")
            print(df.groupby("tiger_count").size())
            # 0 tigers: 482 frames (no tiger)
            # 1 tiger : 310 frames
            # 2 tigers:  47 frames (rare — multiple individuals)
        """
        image_dir = Path(image_dir)
        rows = []
        img_paths = (
            list(image_dir.glob("*.jpg"))
            + list(image_dir.glob("*.png"))
            + list(image_dir.glob("*.JPG"))
        )

        print(f"Processing {len(img_paths)} images from {image_dir}...")

        for i, img_path in enumerate(img_paths, 1):
            if i % 50 == 0:
                print(f"  {i}/{len(img_paths)}")
            try:
                dets = self.detect(str(img_path))
                rows.append({
                    "image_path": str(img_path),
                    "tiger_count": len(dets),
                    "max_confidence": max((d.confidence for d in dets), default=0.0),
                    "bboxes": [(d.bbox, round(d.confidence, 3)) for d in dets],
                })
            except Exception as e:
                print(f"  Error on {img_path.name}: {e}")
                rows.append({
                    "image_path": str(img_path),
                    "tiger_count": -1,
                    "max_confidence": 0.0,
                    "bboxes": [],
                })

        df = pd.DataFrame(rows)
        print(f"\nDone. Tiger count distribution:")
        print(df[df.tiger_count >= 0]["tiger_count"].value_counts().sort_index())
        return df


# ---------------------------------------------------------------------------
# Dataset preparation (for training YOLOv8)
# ---------------------------------------------------------------------------

def prepare_yolo_dataset(
    source_dir: str,
    output_dir: str,
    val_split: float = 0.2,
) -> str:
    """
    Organise annotated images into YOLO training structure and write data.yaml.

    Expects:
        source_dir/
            images/   ← .jpg files
            labels/   ← .txt files (YOLO format: class cx cy w h)

    Creates:
        output_dir/
            images/train/
            images/val/
            labels/train/
            labels/val/
            data.yaml    ← returned path

    Usage:
        yaml_path = prepare_yolo_dataset("data/annotated", "data/yolo")
        train_yolo(yaml_path)
    """
    import random

    src = Path(source_dir)
    out = Path(output_dir)

    images = sorted((src / "images").glob("*.jpg"))
    random.shuffle(images)

    split_idx = int(len(images) * (1 - val_split))
    splits = {"train": images[:split_idx], "val": images[split_idx:]}

    for split, img_list in splits.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

        for img_path in img_list:
            shutil.copy(img_path, out / "images" / split / img_path.name)
            label_path = src / "labels" / (img_path.stem + ".txt")
            if label_path.exists():
                shutil.copy(label_path, out / "labels" / split / label_path.name)
            else:
                # No label = no tiger in this image (create empty label)
                (out / "labels" / split / (img_path.stem + ".txt")).touch()

    yaml_content = f"""path: {out.resolve()}
train: images/train
val: images/val

nc: 1
names:
  0: tiger
"""
    yaml_path = out / "data.yaml"
    yaml_path.write_text(yaml_content)

    print(f"YOLO dataset prepared: {split_idx} train / {len(images)-split_idx} val")
    print(f"data.yaml written: {yaml_path}")
    return str(yaml_path)


def train_yolo(
    data_yaml: str,
    base_model: str = "yolov8n.pt",
    epochs: int = 50,
    img_size: int = 640,
    output_name: str = "tiger_detector",
) -> str:
    """
    Fine-tune YOLOv8 on Sundarban tiger images.
    Returns path to trained weights file.

    Args:
        data_yaml: Path to data.yaml (from prepare_yolo_dataset).
        base_model: YOLOv8 checkpoint to start from.
            "yolov8n.pt" — nano, fast, smaller dataset OK
            "yolov8m.pt" — medium, better accuracy, needs more data
        epochs: Training epochs (50 is usually enough for fine-tuning).
        img_size: Input resolution (640 is YOLO default).
        output_name: Name for the run directory under runs/detect/.

    Usage:
        yaml = prepare_yolo_dataset("data/annotated", "data/yolo")
        weights = train_yolo(yaml, epochs=50)
        detector = TigerDetector(model_path=weights)
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("pip install ultralytics")

    model = YOLO(base_model)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        name=output_name,
        patience=10,          # early stopping
        save=True,
        plots=True,           # saves mAP curves, confusion matrix
    )

    best_weights = Path(f"runs/detect/{output_name}/weights/best.pt")
    if best_weights.exists():
        # Copy to models/ for easy access
        Path("models").mkdir(exist_ok=True)
        dest = Path("models") / "yolo_tiger.pt"
        shutil.copy(best_weights, dest)
        print(f"\nBest weights saved: {dest}")
        return str(dest)

    return str(best_weights)
