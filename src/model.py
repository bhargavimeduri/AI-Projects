"""
model.py
Model architecture definitions for Sundarban Tiger project.
"""

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
