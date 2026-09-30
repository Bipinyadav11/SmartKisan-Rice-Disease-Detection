import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ============================================================
# SMARTKISAN - IMPROVED RICE DISEASE MODEL (v2 - FIXED)
# ============================================================
# Changes vs old version:
#   1. class_weight added (handles imbalance across 6 classes)
#   2. EarlyStopping + ReduceLROnPlateau added
#   3. TWO-PHASE TRAINING:
#        Phase 1 -> train only the head (frozen base), few epochs
#        Phase 2 -> unfreeze last ~50 layers of MobileNetV2 and
#                   fine-tune with a very low learning rate
#      This lets the network actually learn rice-leaf-specific
#      textures instead of relying only on generic ImageNet
#      features, which is why similar diseases (brown_spot vs
#      narrow_brown_spot, etc.) were getting confused before.
# ============================================================

TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/validation"

MODEL_NAME = "rice_disease_model.h5"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

PHASE1_EPOCHS = 15   # head-only training
PHASE2_EPOCHS = 25   # fine-tuning (early stopping will cut this short if needed)

FINE_TUNE_LAYERS = 50  # how many layers from the END of MobileNetV2 to unfreeze


# ============================================================
# EXACT DISEASE CLASS ORDER
# ============================================================

class_names = [
    "bacterial_leaf_blight",
    "brown_spot",
    "healthy",
    "leaf_blast",
    "leaf_scald",
    "narrow_brown_spot"
]


# ============================================================
# TRAINING DATA
# ============================================================

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.20,
    shear_range=0.15,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    classes=class_names,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

val_data = val_datagen.flow_from_directory(
    VAL_DIR,
    classes=class_names,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

print()
print("======================================")
print("DISEASE CLASS MAPPING")
print("======================================")
print(train_data.class_indices)
print()


# ============================================================
# CLASS WEIGHTS (handles imbalance across the 6 classes)
# ============================================================

counts = np.bincount(train_data.classes)
total = counts.sum()
n_classes = len(counts)

class_weight = {
    i: (total / (n_classes * count)) if count > 0 else 1.0
    for i, count in enumerate(counts)
}

print("======================================")
print("IMAGES PER CLASS")
print("======================================")
for i, name in enumerate(class_names):
    print(f"{i} = {name} : {counts[i]} images")
print()
print("Calculated Class Weights:", class_weight)
print()


# ============================================================
# BUILD MODEL (MobileNetV2 base)
# ============================================================

print("Loading MobileNetV2...")

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False  # frozen for Phase 1

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(len(class_names), activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

callbacks_phase1 = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)
]

print()
print("======================================")
print("PHASE 1: TRAINING HEAD ONLY (base frozen)")
print("======================================")
print()

history1 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=PHASE1_EPOCHS,
    class_weight=class_weight,
    callbacks=callbacks_phase1
)


# ============================================================
# PHASE 2: FINE-TUNE LAST LAYERS OF MOBILENETV2
# ============================================================

print()
print("======================================")
print(f"PHASE 2: FINE-TUNING LAST {FINE_TUNE_LAYERS} LAYERS")
print("======================================")
print()

base_model.trainable = True

# Freeze everything except the last FINE_TUNE_LAYERS layers
for layer in base_model.layers[:-FINE_TUNE_LAYERS]:
    layer.trainable = False

# Recompile with a MUCH lower learning rate - critical when
# unfreezing pretrained weights, otherwise you destroy the
# pretrained features instead of gently adapting them.
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_phase2 = [
    EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7)
]

history2 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=PHASE2_EPOCHS,
    class_weight=class_weight,
    callbacks=callbacks_phase2
)


# ============================================================
# SAVE MODEL
# ============================================================

model.save(MODEL_NAME)

print()
print("======================================")
print("DISEASE TRAINING FINISHED")
print("======================================")
print()
print("New model saved as:", MODEL_NAME)
print()
print("Classes used:")
for i, name in enumerate(class_names):
    print(i, "=", name)
print()
print("IMPORTANT:")
print("rice_vs_invalid_model.h5 was NOT changed.")