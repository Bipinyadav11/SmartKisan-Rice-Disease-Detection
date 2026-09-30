import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import traceback
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ============================================================
# SMARTKISAN - STABLE DUAL-STAGE INFERENCE ENGINE (v2 - TTA)
# ============================================================
# WHY THIS VERSION IS MORE RELIABLE ON UNSEEN (Google/phone) IMAGES:
#
# A single prediction from ONE crop/lighting version of an image
# can be misleadingly confident (e.g. 96% wrong). This happens
# because the model reacts to a specific pixel pattern in that
# ONE version of the image.
#
# Test-Time Augmentation (TTA) generates several slightly
# different versions of the SAME input image (flipped, brighter,
# darker, slightly zoomed) and averages the model's predictions
# across all of them. This smooths out one-off mistakes and gives
# a much more trustworthy confidence score - it is one of the
# most effective, well-established techniques for exactly this
# problem, and does not require retraining or new data.
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BINARY_MODEL_PATH = os.path.join(BASE_DIR, "rice_vs_invalid_model.h5")
DISEASE_MODEL_PATH = os.path.join(BASE_DIR, "rice_disease_model.h5")

CLASS_NAMES = [
    "bacterial_leaf_blight",
    "brown_spot",
    "healthy",
    "leaf_blast",
    "leaf_scald",
    "narrow_brown_spot"
]

def exit_invalid(msg="Invalid Image", conf=0.0):
    print(f"Disease : {msg}")
    print(f"Confidence : {round(conf, 2)} %")
    sys.exit()

# ------------------------------------------------------------
# 1. READ ARGUMENTS & VALIDATE FILE
# ------------------------------------------------------------
if len(sys.argv) < 2:
    exit_invalid()

img_path = sys.argv[1]

if not os.path.isfile(img_path):
    exit_invalid()

# ------------------------------------------------------------
# 2. IMAGE PREPROCESSING (base version)
# ------------------------------------------------------------
try:
    img = cv2.imread(img_path)
    if img is None:
        exit_invalid()

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224)).astype(np.float32)

except Exception:
    print("---- DEBUG: CRASH IN IMAGE PREPROCESSING ----", file=sys.stderr)
    traceback.print_exc()
    exit_invalid()


# ------------------------------------------------------------
# 3. BUILD TTA VERSIONS OF THE IMAGE
# ------------------------------------------------------------
def make_tta_batch(base_img):
    versions = []

    # 1. Original
    versions.append(base_img)

    # 2. Horizontal flip
    versions.append(cv2.flip(base_img, 1))

    # 3. Brighter
    bright = np.clip(base_img * 1.2, 0, 255)
    versions.append(bright)

    # 4. Darker
    dark = np.clip(base_img * 0.8, 0, 255)
    versions.append(dark)

    # 5. Slight center zoom-in (crop 90% then resize back)
    h, w = base_img.shape[:2]
    ch, cw = int(h * 0.9), int(w * 0.9)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    cropped = base_img[y0:y0 + ch, x0:x0 + cw]
    zoomed = cv2.resize(cropped, (w, h))
    versions.append(zoomed)

    batch = np.stack(versions, axis=0)
    return preprocess_input(batch)


tta_batch = make_tta_batch(img_resized)


# ------------------------------------------------------------
# 4. STAGE 1: IS IT A RICE LEAF? (Binary Check, TTA-averaged)
# ------------------------------------------------------------
if os.path.exists(BINARY_MODEL_PATH):
    try:
        binary_model = tf.keras.models.load_model(BINARY_MODEL_PATH, compile=False)
        binary_preds = binary_model.predict(tta_batch, verbose=0)[:, 0]
        binary_pred = float(np.mean(binary_preds))

        # Binary mapping assumed: 0 = Invalid, 1 = Rice
        if binary_pred < 0.40:
            invalid_confidence = (1.0 - binary_pred) * 100.0
            exit_invalid("Invalid Image", invalid_confidence)
    except Exception:
        pass  # fall back gracefully to Stage 2 if binary model fails

# ------------------------------------------------------------
# 5. STAGE 2: WHICH DISEASE IS IT? (Multi-Class, TTA-averaged)
# ------------------------------------------------------------
try:
    disease_model = tf.keras.models.load_model(DISEASE_MODEL_PATH, compile=False)
    tta_predictions = disease_model.predict(tta_batch, verbose=0)

    # Average across all TTA versions -> much more stable than
    # trusting a single crop/lighting version of the image.
    predictions = np.mean(tta_predictions, axis=0)

    # How much the TTA versions agree with each other is itself
    # useful information: if they disagree a lot, the model is
    # genuinely unsure about this image.
    per_version_top = np.argmax(tta_predictions, axis=1)
    agreement = np.mean(per_version_top == np.argmax(predictions))

    sorted_indices = np.argsort(predictions)[::-1]
    top1_idx = sorted_indices[0]
    top2_idx = sorted_indices[1]

    top1_score = predictions[top1_idx] * 100.0
    top2_score = predictions[top2_idx] * 100.0
    margin = top1_score - top2_score

    # --------------------------------------------------------
    # DIAGNOSTIC MODE: run with an extra "-debug" argument to
    # always see the top-3 raw scores, even when the image gets
    # rejected as invalid. This is the only way to tell whether
    # the model is "confused between 2 diseases" (both classes
    # show decent scores) vs "sees nothing familiar at all"
    # (all 6 scores are low/flat, ~16% each = random guessing).
    # --------------------------------------------------------
    if "-debug" in sys.argv:
        print("---- DEBUG: TOP 3 RAW SCORES ----", file=sys.stderr)
        for idx in sorted_indices[:3]:
            print(f"{CLASS_NAMES[idx]} : {round(predictions[idx] * 100.0, 2)} %", file=sys.stderr)
        print(f"TTA agreement : {round(agreement * 100.0, 1)} %", file=sys.stderr)
        print("----------------------------------", file=sys.stderr)

    # SAFETY CHECK (relaxed thresholds + TTA agreement):
    # 1. Absolute top score too low -> model doesn't know what it is.
    # 2. Margin between #1 and #2 too small -> model is confused
    #    between two specific diseases.
    # 3. TTA versions disagree with each other -> unstable prediction.
    if top1_score < 35.0 or margin < 10.0 or agreement < 0.5:
        exit_invalid("Invalid Image", 100.0 - top1_score)

    predicted_disease = CLASS_NAMES[top1_idx]

    print(f"Disease : {predicted_disease}")
    print(f"Confidence : {round(top1_score, 2)} %")

except Exception:
    print("---- DEBUG: CRASH IN DISEASE STAGE ----", file=sys.stderr)
    traceback.print_exc()
    exit_invalid()