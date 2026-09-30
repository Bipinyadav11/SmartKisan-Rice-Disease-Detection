import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ============================================================
# SMARTKISAN - RICE VS INVALID OPTIMIZED TRAINING (STEP 1)
# ============================================================

TRAIN_DIR = "binary_dataset/train"
VAL_DIR = "binary_dataset/validation"
MODEL_PATH = "rice_vs_invalid_model.h5"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20

# Data Augmentation with MobileNet Preprocessing
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.25,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest"
)

validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

print("\n======================================")
print("Loading Binary Training Dataset (Step 1)")
print("======================================")

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True
)

validation_data = validation_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

print("\nClass Mapping:", train_data.class_indices)

# Automatically calculate balanced class weights to handle 56 items vs 350+ items
invalid_count = 0
rice_count = 0

for class_name, class_index in train_data.class_indices.items():
    if class_name.lower() == "invalid":
        invalid_count = sum(1 for label in train_data.classes if label == class_index)
    elif class_name.lower() == "rice":
        rice_count = sum(1 for label in train_data.classes if label == class_index)

total = invalid_count + rice_count
class_weight = {
    0: (total / (2.0 * invalid_count)) if invalid_count > 0 else 1.0,
    1: (total / (2.0 * rice_count)) if rice_count > 0 else 1.0
}

print("Calculated Class Weights:", class_weight)

# Build Model using MobileNetV2 Base
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False

model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)
]

print("\n======================================")
print("STARTING STEP 1 BINARY MODEL TRAINING")
print("======================================\n")

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=EPOCHS,
    class_weight=class_weight,
    callbacks=callbacks
)

model.save(MODEL_PATH)

print("\n======================================")
print("STEP 1 TRAINING FINISHED")
print(f"New model saved as: {MODEL_PATH}")
print("======================================")