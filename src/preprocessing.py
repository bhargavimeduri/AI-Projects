"""
preprocessing.py
Image preprocessing + data augmentation for Sundarban Tiger project.
"""

import cv2
import numpy as np
from pathlib import Path
import albumentations as A


IMG_SIZE = (224, 224)  # Standard input size for ResNet50 / EfficientNetB3


# ---------------------------------------------------------------------------
# Augmentation pipelines
# ---------------------------------------------------------------------------

# Training pipeline — applies random transforms to multiply dataset size
AUGMENTATION_PIPELINE_TRAIN = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
    A.Rotate(limit=15, p=0.5),
    A.MotionBlur(blur_limit=7, p=0.3),           # mimics camera blur in jungle
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3), # mimics low-light sensor noise
    A.CoarseDropout(
        max_holes=8, max_height=20, max_width=20, fill_value=0, p=0.3
    ),                                             # occlusion by leaves/branches
    A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
])

# Validation / inference pipeline — only resize, no random transforms
AUGMENTATION_PIPELINE_VAL = A.Compose([
    A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
])


# ---------------------------------------------------------------------------
# Core image utilities
# ---------------------------------------------------------------------------

def load_image(image_path: str) -> np.ndarray:
    """Load image from disk as RGB numpy array."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_image(img: np.ndarray, size: tuple = IMG_SIZE) -> np.ndarray:
    """Resize image to target size."""
    return cv2.resize(img, size)


def normalise_image(img: np.ndarray) -> np.ndarray:
    """Normalise pixel values to [0, 1]."""
    return img.astype(np.float32) / 255.0


def enhance_contrast(img: np.ndarray) -> np.ndarray:
    """
    Enhance contrast using CLAHE — vital for dark camera trap images.
    CLAHE = Contrast Limited Adaptive Histogram Equalisation.
    Operates on the L (lightness) channel only — colours stay unchanged.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def preprocess_pipeline(image_path: str, training: bool = False) -> np.ndarray:
    """
    Full preprocessing pipeline for a single image.
    training=True applies random augmentations (for training data).
    training=False applies only resize (for val/inference).
    Returns float32 array normalised to [0, 1].
    """
    img = load_image(image_path)
    img = enhance_contrast(img)
    pipeline = AUGMENTATION_PIPELINE_TRAIN if training else AUGMENTATION_PIPELINE_VAL
    img = pipeline(image=img)["image"]
    return normalise_image(img)


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------

def augment_image(img: np.ndarray, n_augments: int = 5) -> list[np.ndarray]:
    """
    Apply the training augmentation pipeline N times to one image.
    Returns list of N augmented variants (for Notebook 02 visualisation).
    """
    return [
        AUGMENTATION_PIPELINE_TRAIN(image=img)["image"]
        for _ in range(n_augments)
    ]


def build_augmented_dataset(
    data_dir: str,
    augment_factor: int = 5,
    img_size: tuple = IMG_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load all images and multiply training set by augment_factor.

    Directory structure expected:
        data_dir/
            tiger/
            no_tiger/

    Returns X (images, float32, normalised) and y (labels: 1=tiger, 0=no_tiger).
    Each real image generates `augment_factor` augmented versions, so a dataset
    of 200 images becomes 200 * augment_factor = 1000 training samples.
    """
    images, labels = [], []
    label_map = {"tiger": 1, "no_tiger": 0}

    for class_name, label in label_map.items():
        class_dir = Path(data_dir) / class_name
        if not class_dir.exists():
            print(f"Warning: {class_dir} not found — skipping")
            continue

        img_paths = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        print(f"  {class_name}: {len(img_paths)} source images → "
              f"{len(img_paths) * augment_factor} augmented")

        for img_path in img_paths:
            try:
                img = load_image(str(img_path))
                img = enhance_contrast(img)
                for aug_img in augment_image(img, n_augments=augment_factor):
                    images.append(normalise_image(aug_img))
                    labels.append(label)
            except Exception as e:
                print(f"  Error processing {img_path.name}: {e}")

    return np.array(images, dtype=np.float32), np.array(labels)


def build_dataset(data_dir: str, img_size: tuple = IMG_SIZE):
    """
    Load images WITHOUT augmentation — for validation / test sets.

    Directory structure:
        data_dir/
            tiger/
            no_tiger/
    """
    images, labels = [], []
    label_map = {"tiger": 1, "no_tiger": 0}

    for class_name, label in label_map.items():
        class_dir = Path(data_dir) / class_name
        if not class_dir.exists():
            print(f"Warning: {class_dir} not found — skipping")
            continue
        for img_path in list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")):
            try:
                img = preprocess_pipeline(str(img_path), training=False)
                images.append(img)
                labels.append(label)
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")

    return np.array(images, dtype=np.float32), np.array(labels)


# ---------------------------------------------------------------------------
# YOLO dataset helper
# ---------------------------------------------------------------------------

def prepare_yolo_crops(
    image_dir: str,
    label_dir: str,
    output_dir: str,
) -> None:
    """
    Extract tiger crops from YOLO-annotated images for the ID model.
    Reads YOLO .txt label files (class cx cy w h, normalised 0-1).
    Saves each detected tiger as a separate crop in output_dir/tiger/.

    Use this AFTER you have YOLO annotations on your raw images.
    """
    output_path = Path(output_dir) / "tiger"
    output_path.mkdir(parents=True, exist_ok=True)

    img_paths = list(Path(image_dir).glob("*.jpg")) + list(Path(image_dir).glob("*.png"))
    crop_count = 0

    for img_path in img_paths:
        label_path = Path(label_dir) / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        with open(label_path) as f:
            for i, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                _, cx, cy, bw, bh = map(float, parts)
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                crop = img[max(0, y1):y2, max(0, x1):x2]
                if crop.size == 0:
                    continue
                out_file = output_path / f"{img_path.stem}_crop{i}.jpg"
                cv2.imwrite(str(out_file), crop)
                crop_count += 1

    print(f"Saved {crop_count} tiger crops to {output_path}")
