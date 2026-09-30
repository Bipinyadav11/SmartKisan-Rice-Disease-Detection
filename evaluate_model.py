import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix

# ============================================================
# SMARTKISAN - MODEL EVALUATION & METRICS SCRIPT
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(BASE_DIR, "dataset", "validation")  # ya "dataset/test" agar alag test folder hai
MODEL_PATH = os.path.join(BASE_DIR, "rice_disease_model.h5")

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# Strict 6 Classes (Alphabetical Order matching Keras flow)
CLASS_NAMES = [
    "bacterial_leaf_blight",
    "brown_spot",
    "healthy",
    "leaf_blast",
    "leaf_scald",
    "narrow_brown_spot"
]

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file not found at: {MODEL_PATH}")
        return

    if not os.path.exists(TEST_DIR):
        print(f"[ERROR] Test/Validation directory not found at: {TEST_DIR}")
        return

    print("Loading Trained Model...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    print("Preparing Test Dataset Generator...")
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    test_data = test_datagen.flow_from_directory(
        TEST_DIR,
        classes=CLASS_NAMES,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False  # Must be False for exact Confusion Matrix evaluation
    )

    print("\nEvaluating Model Predictions...")
    # Get raw prediction probabilities
    predictions = model.predict(test_data, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_data.classes

    # Calculate overall Accuracy
    accuracy = np.mean(y_pred == y_true) * 100
    print("\n========================================================")
    print(f"  TOTAL OVERALL ACCURACY : {round(accuracy, 2)} %")
    print("========================================================\n")

    # Detailed Per-Class Classification Report
    print("CLASSIFICATION REPORT:\n")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

    # Print Confusion Matrix
    print("CONFUSION MATRIX:\n")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    print("\n========================================================")

if __name__ == "__main__":
    main()