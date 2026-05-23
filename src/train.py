"""
train.py
Production training script for Sundarban Tiger project.
Tracks all experiments with MLflow.

Usage:
    python src/train.py --stage detection --epochs 30 --data data/processed
    python src/train.py --stage identification --epochs 50 --num_tigers 50
"""

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import tensorflow as tf
import mlflow
import mlflow.tensorflow

from model import build_detection_model, build_identification_model, unfreeze_and_finetune
from preprocessing import build_dataset, build_augmented_dataset


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """All hyperparameters in one place — easy to log and reproduce."""
    stage: str          = "detection"     # "detection" or "identification"
    data_dir: str       = "data/processed"
    model_dir: str      = "models"
    epochs_phase1: int  = 20              # frozen base layers
    epochs_phase2: int  = 10              # unfrozen fine-tuning
    batch_size: int     = 32
    img_size: tuple     = (224, 224)
    augment_factor: int = 5               # each image → 5 augmented copies
    num_tigers: int     = 50              # for identification stage
    learning_rate: float = 1e-4
    experiment_name: str = "sundarban_tiger_enumeration"


# ---------------------------------------------------------------------------
# MLflow setup
# ---------------------------------------------------------------------------

def setup_mlflow(config: TrainingConfig) -> str:
    """Initialise MLflow experiment. Returns run_id."""
    mlflow.set_experiment(config.experiment_name)
    return mlflow.start_run(run_name=f"{config.stage}_{int(time.time())}")


# ---------------------------------------------------------------------------
# Detection training (Stage 1 — ResNet50)
# ---------------------------------------------------------------------------

def train_detection_model(config: TrainingConfig) -> None:
    """
    Two-phase training for ResNet50 tiger detection.

    Phase 1 (Frozen):
        Base frozen → only top Dense layers train.
        High learning rate (1e-4).
        Fast convergence — learning domain basics.

    Phase 2 (Fine-tuning):
        Top 30 ResNet50 layers unfrozen.
        Low learning rate (1e-5) — avoid destroying learned features.
        Squeezes out extra accuracy.
    """
    print("\n" + "="*60)
    print("STAGE 1: Tiger Detection (ResNet50)")
    print("="*60)

    # Load data
    train_dir = Path(config.data_dir) / "train"
    val_dir   = Path(config.data_dir) / "val"

    print("\nLoading training data with augmentation...")
    X_train, y_train = build_augmented_dataset(
        str(train_dir), augment_factor=config.augment_factor
    )
    print("Loading validation data...")
    X_val, y_val = build_dataset(str(val_dir))

    print(f"\nTrain: {len(X_train)} images | Val: {len(X_val)} images")
    print(f"Class balance — Train: {y_train.mean():.2%} tiger")

    model = build_detection_model()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", patience=5, restore_best_weights=True, mode="max"
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f"{config.model_dir}/detection_best.h5",
            monitor="val_auc", save_best_only=True, mode="max"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7
        ),
    ]

    with mlflow.start_run(run_name="detection_phase1") as run:
        mlflow.log_params(asdict(config))
        mlflow.log_param("phase", 1)

        print("\n--- Phase 1: Frozen base ---")
        history1 = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=config.epochs_phase1,
            batch_size=config.batch_size,
            callbacks=callbacks,
        )
        _log_history(history1, prefix="phase1")

        print("\n--- Phase 2: Fine-tuning top 30 layers ---")
        model = unfreeze_and_finetune(model, num_layers=30)
        history2 = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=config.epochs_phase2,
            batch_size=config.batch_size,
            callbacks=callbacks,
        )
        _log_history(history2, prefix="phase2")

        # Final metrics
        val_loss, val_acc, val_auc = model.evaluate(X_val, y_val, verbose=0)
        mlflow.log_metrics({
            "final_val_accuracy": val_acc,
            "final_val_auc": val_auc,
            "final_val_loss": val_loss,
        })

        # False Negative Rate — project target: < 5%
        y_pred = (model.predict(X_val) > 0.5).astype(int).flatten()
        tiger_mask = y_val == 1
        fnr = 1 - (y_pred[tiger_mask] == 1).mean() if tiger_mask.any() else 0.0
        mlflow.log_metric("false_negative_rate", fnr)

        print(f"\n{'='*40}")
        print(f"Detection Results:")
        print(f"  Val Accuracy : {val_acc:.4f}  (target: >0.90)")
        print(f"  Val AUC      : {val_auc:.4f}")
        print(f"  FNR          : {fnr:.4f}  (target: <0.05)")
        print(f"{'='*40}")

        # Save final model
        Path(config.model_dir).mkdir(parents=True, exist_ok=True)
        model.save(f"{config.model_dir}/detection_final.h5")
        mlflow.tensorflow.log_model(model, "detection_model")
        print(f"\nModel saved: {config.model_dir}/detection_final.h5")
        print(f"MLflow run ID: {run.info.run_id}")


# ---------------------------------------------------------------------------
# Identification training (Stage 2 — EfficientNetB3)
# ---------------------------------------------------------------------------

def train_identification_model(config: TrainingConfig) -> None:
    """
    Train EfficientNetB3 to identify individual tigers by stripe pattern.
    Expects data_dir/train/T-{01..N}/ folder per tiger.
    """
    print("\n" + "="*60)
    print("STAGE 2: Individual Tiger Identification (EfficientNetB3)")
    print("="*60)

    train_dir = Path(config.data_dir) / "train"
    val_dir   = Path(config.data_dir) / "val"

    # Use Keras ImageDataGenerator for multi-class ID data
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    datagen_train = ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=True,
        rotation_range=15,
        brightness_range=[0.7, 1.3],
        zoom_range=0.1,
    )
    datagen_val = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = datagen_train.flow_from_directory(
        str(train_dir),
        target_size=(300, 300),
        batch_size=config.batch_size,
        class_mode="categorical",
    )
    val_gen = datagen_val.flow_from_directory(
        str(val_dir),
        target_size=(300, 300),
        batch_size=config.batch_size,
        class_mode="categorical",
    )

    num_classes = train_gen.num_classes
    print(f"Found {num_classes} tiger classes")

    model = build_identification_model(num_tigers=num_classes)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=7, restore_best_weights=True, mode="max"
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f"{config.model_dir}/identification_best.h5",
            monitor="val_accuracy", save_best_only=True, mode="max"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7
        ),
    ]

    with mlflow.start_run(run_name="identification") as run:
        mlflow.log_params(asdict(config))
        mlflow.log_param("num_tiger_classes", num_classes)

        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=config.epochs_phase1 + config.epochs_phase2,
            callbacks=callbacks,
        )
        _log_history(history, prefix="id")

        val_loss, val_acc = model.evaluate(val_gen, verbose=0)
        mlflow.log_metrics({
            "final_val_accuracy": val_acc,
            "final_val_loss": val_loss,
        })

        print(f"\n{'='*40}")
        print(f"Identification Results:")
        print(f"  Val Accuracy : {val_acc:.4f}  (target: >0.80)")
        print(f"{'='*40}")

        Path(config.model_dir).mkdir(parents=True, exist_ok=True)
        model.save(f"{config.model_dir}/identification_final.h5")

        # Save class index mapping (T-17 → index 4, etc.)
        class_map = {v: k for k, v in train_gen.class_indices.items()}
        with open(f"{config.model_dir}/tiger_class_map.json", "w") as f:
            json.dump(class_map, f, indent=2)

        mlflow.tensorflow.log_model(model, "identification_model")
        print(f"\nModel saved: {config.model_dir}/identification_final.h5")
        print(f"Class map saved: {config.model_dir}/tiger_class_map.json")
        print(f"MLflow run ID: {run.info.run_id}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_history(history, prefix: str) -> None:
    """Log epoch-level metrics to MLflow."""
    for epoch, (loss, val_loss) in enumerate(
        zip(history.history.get("loss", []),
            history.history.get("val_loss", []))
    ):
        mlflow.log_metrics(
            {f"{prefix}_loss": loss, f"{prefix}_val_loss": val_loss},
            step=epoch
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train tiger detection or identification model"
    )
    parser.add_argument(
        "--stage", choices=["detection", "identification"],
        default="detection", help="Which model to train"
    )
    parser.add_argument("--data", default="data/processed", help="Path to processed data dir")
    parser.add_argument("--epochs", type=int, default=None, help="Override total epochs")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_tigers", type=int, default=50,
                        help="Number of individual tigers (identification only)")
    parser.add_argument("--augment_factor", type=int, default=5,
                        help="Augmentation multiplier per training image")
    args = parser.parse_args()

    config = TrainingConfig(
        stage=args.stage,
        data_dir=args.data,
        batch_size=args.batch_size,
        num_tigers=args.num_tigers,
        augment_factor=args.augment_factor,
    )
    if args.epochs is not None:
        config.epochs_phase1 = max(1, args.epochs - 10)
        config.epochs_phase2 = min(10, args.epochs)

    print(f"\nSundarban Tiger Project — Training Script")
    print(f"Stage      : {config.stage}")
    print(f"Data dir   : {config.data_dir}")
    print(f"Batch size : {config.batch_size}")
    print(f"MLflow exp : {config.experiment_name}")

    if config.stage == "detection":
        train_detection_model(config)
    else:
        train_identification_model(config)


if __name__ == "__main__":
    main()
