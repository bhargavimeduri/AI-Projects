"""
model.py
Model architecture definitions for Sundarban Tiger project.
Includes: detection (ResNet50), identification (EfficientNetB3),
          embedding extraction, and cosine similarity for open-set ID.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50, EfficientNetB3


def build_detection_model(num_classes: int = 1) -> Model:
    """
    Stage 1 — Tiger Detection Model.
    Binary classifier: Tiger (1) vs No Tiger (0).
    Uses ResNet50 as backbone with transfer learning.
    """
    base = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )
    # Freeze base layers — use ImageNet features as-is
    base.trainable = False

    inputs  = tf.keras.Input(shape=(224, 224, 3))
    x       = base(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.Dense(256, activation="relu")(x)
    x       = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs, name="tiger_detection")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model


def build_identification_model(num_tigers: int) -> Model:
    """
    Stage 2 — Individual Tiger Identification Model.
    Multi-class classifier: Which tiger is this?
    Uses EfficientNetB3 with fine-tuning.
    """
    base = EfficientNetB3(
        weights="imagenet",
        include_top=False,
        input_shape=(300, 300, 3)
    )
    # Fine-tune last 20 layers
    for layer in base.layers[:-20]:
        layer.trainable = False

    inputs  = tf.keras.Input(shape=(300, 300, 3))
    x       = base(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.Dense(512, activation="relu")(x)
    x       = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_tigers, activation="softmax")(x)

    model = Model(inputs, outputs, name="tiger_identification")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def unfreeze_and_finetune(model: Model, num_layers: int = 30) -> Model:
    """
    Unfreeze top N layers of base model for fine-tuning.
    Call this after initial training converges.
    """
    base = model.layers[1]  # base model is second layer
    base.trainable = True
    for layer in base.layers[:-num_layers]:
        layer.trainable = False

    # Lower learning rate for fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=model.loss,
        metrics=model.metrics
    )
    return model


# ---------------------------------------------------------------------------
# Embedding & similarity — for open-set individual tiger identification
# ---------------------------------------------------------------------------

def build_embedding_model(identification_model: Model) -> Model:
    """
    Extract the penultimate Dense layer from the trained identification model
    as a feature extractor.

    The result is a 512-dimensional embedding vector for each tiger image.
    Two images of the SAME tiger → high cosine similarity (> 0.85).
    Two images of DIFFERENT tigers → low cosine similarity (< 0.5).

    Usage:
        id_model = build_identification_model(num_tigers=50)
        id_model.load_weights("models/tiger_id.h5")
        emb_model = build_embedding_model(id_model)
        vec = emb_model.predict(image_batch)   # shape: (batch, 512)
    """
    # Cut the model just before the final softmax layer
    # Layer structure: ... → Dense(512, relu) → Dropout → Dense(N, softmax)
    # We want the Dense(512) output
    embedding_layer = None
    for layer in reversed(identification_model.layers):
        if isinstance(layer, layers.Dense) and layer.units == 512:
            embedding_layer = layer
            break

    if embedding_layer is None:
        raise ValueError(
            "Could not find Dense(512) layer in identification model. "
            "Make sure you pass the unmodified model from build_identification_model()."
        )

    return Model(
        inputs=identification_model.input,
        outputs=embedding_layer.output,
        name="tiger_embedding"
    )


def compute_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Cosine similarity between two 512-dim embedding vectors.
    Returns a float in [0, 1] — higher means more similar.

    Rule of thumb for tiger ID:
        > 0.85  → very likely the same individual
        0.6–0.85 → possibly the same — flag for human review
        < 0.6   → different tigers

    Example:
        emb_T17_cam1 = emb_model.predict(img_cam1)[0]
        emb_T17_cam2 = emb_model.predict(img_cam2)[0]
        score = compute_similarity(emb_T17_cam1, emb_T17_cam2)
        # → 0.91 (same tiger, different camera trap)
    """
    e1 = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
    e2 = embedding2 / (np.linalg.norm(embedding2) + 1e-8)
    return float(np.dot(e1, e2))


def find_closest_tiger(
    query_embedding: np.ndarray,
    gallery_embeddings: dict[str, np.ndarray],
    threshold: float = 0.75,
) -> tuple[str | None, float]:
    """
    Open-set identification: given a new tiger image embedding,
    find the closest known individual in the gallery.

    Args:
        query_embedding: 512-dim vector for the unknown tiger.
        gallery_embeddings: dict mapping tiger_id → embedding vector.
            e.g. {"T-17": array([...]), "T-23": array([...])}
        threshold: minimum similarity to declare a match (default 0.75).
            Below threshold → tiger is declared "NEW / UNKNOWN".

    Returns:
        (tiger_id, similarity_score) — tiger_id is None if below threshold.

    Usage:
        gallery = {tid: emb_model.predict(ref_img)[0] for tid, ref_img in refs}
        tiger_id, score = find_closest_tiger(query_emb, gallery)
        if tiger_id:
            print(f"Matched: {tiger_id} ({score:.2f})")
        else:
            print("New tiger — adding to gallery")
    """
    best_id, best_score = None, -1.0
    for tiger_id, ref_emb in gallery_embeddings.items():
        score = compute_similarity(query_embedding, ref_emb)
        if score > best_score:
            best_score = score
            best_id = tiger_id

    if best_score >= threshold:
        return best_id, best_score
    return None, best_score
