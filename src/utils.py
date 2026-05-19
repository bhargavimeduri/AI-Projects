"""
utils.py
Helper functions — plotting, metrics, saving results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import json
from pathlib import Path


def plot_training_history(history, save_path: str = None):
    """Plot accuracy and loss curves from model.fit() history."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history["accuracy"],     label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    # Loss
    axes[1].plot(history.history["loss"],     label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names: list, save_path: str = None):
    """Plot confusion matrix with labels."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))


def save_results(results: dict, output_path: str):
    """Save evaluation results to JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {output_path}")


def show_sample_predictions(images, y_true, y_pred, class_names: list, n: int = 9):
    """Display sample images with predicted vs actual labels."""
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    indices = np.random.choice(len(images), n, replace=False)

    for i, idx in enumerate(indices):
        ax = axes[i // 3][i % 3]
        ax.imshow(images[idx])
        actual    = class_names[y_true[idx]]
        predicted = class_names[y_pred[idx]]
        colour    = "green" if actual == predicted else "red"
        ax.set_title(f"Actual: {actual}\nPred: {predicted}", color=colour)
        ax.axis("off")

    plt.suptitle("Sample Predictions (Green=Correct, Red=Wrong)", fontsize=14)
    plt.tight_layout()
    plt.show()
