import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# SMARTKISAN - RICE VS INVALID MODEL EVALUATION
# ============================================================

VAL_DIR = "binary_dataset/validation"
MODEL_PATH = "rice_vs_invalid_model.h5"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print()
print("======================================")
print("SMARTKISAN BINARY MODEL EVALUATION")
print("======================================")

validation_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0
)

validation_data = validation_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

print()
print("Validation images:", validation_data.samples)
print("Class mapping:", validation_data.class_indices)


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("Loading binary model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Binary model loaded successfully.")


# ============================================================
# PREDICTIONS
# ============================================================

print()
print("Running predictions...")
print("Please wait...")

predictions = model.predict(
    validation_data,
    verbose=1
)

# Sigmoid output
predicted_classes = (
    predictions[:, 0] >= 0.5
).astype(int)

actual_classes = validation_data.classes


# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(
    actual_classes,
    predicted_classes
)

print()
print("======================================")
print("OVERALL ACCURACY")
print("======================================")

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("======================================")
print("CLASSIFICATION REPORT")
print("======================================")

print(
    classification_report(
        actual_classes,
        predicted_classes,
        target_names=["Invalid", "Rice"],
        digits=4
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print()
print("======================================")
print("CONFUSION MATRIX")
print("======================================")

print(
    confusion_matrix(
        actual_classes,
        predicted_classes
    )
)


# ============================================================
# FINISHED
# ============================================================

print()
print("======================================")
print("BINARY EVALUATION FINISHED")
print("======================================")