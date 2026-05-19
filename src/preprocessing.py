"""
preprocessing.py
Image preprocessing utilities for Sundarban Tiger project.
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image


IMG_SIZE = (224, 224)  # Standard input size for ResNet50


def load_image(image_path: str) -> np.ndarray:
    """Load image from disk as RGB numpy array."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def resize_image(img: np.ndarray, size: tuple = IMG_SIZE) -> np.ndarray:
    """Resize image to target size."""
    return cv2.resize(img, size)


def normalise_image(img: np.ndarray) -> np.ndarray:
    """Normalise pixel values to [0, 1]."""
    return img.astype(np.float32) / 255.0


def enhance_contrast(img: np.ndarray) -> np.ndarray:
    """
    Enhance contrast using CLAHE — useful for dark camera trap images.
    CLAHE = Contrast Limited Adaptive Histogram Equalisation.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def preprocess_pipeline(image_path: str) -> np.ndarray:
    """
    Full preprocessing pipeline for a single image.
    Run this on every image before feeding into the model.
    """
    img = load_image(image_path)
    img = enhance_contrast(img)
    img = resize_image(img)
    img = normalise_image(img)
    return img


def build_dataset(data_dir: str, img_size: tuple = IMG_SIZE):
    """
    Load all images from a directory structure:
      data_dir/
        tiger/
        no_tiger/
    Returns X (images) and y (labels).
    """
    images, labels = [], []
    label_map = {"tiger": 1, "no_tiger": 0}

    for class_name, label in label_map.items():
        class_dir = Path(data_dir) / class_name
        if not class_dir.exists():
            print(f"Warning: {class_dir} not found — skipping")
            continue
        for img_path in class_dir.glob("*.jpg"):
            try:
                img = preprocess_pipeline(str(img_path))
                images.append(img)
                labels.append(label)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    return np.array(images), np.array(labels)
