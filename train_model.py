import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ----------------------------
# CONFIG
# ----------------------------
DATASET_DIR = "footwear"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 5

# ----------------------------
# DATA GENERATOR
# ----------------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",   # 🔥 FIX
    subset="training"
)

val_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",   # 🔥 FIX
    subset="validation"
)

num_classes = train_data.num_classes
print("Class mapping:", train_data.class_indices)

# ----------------------------
# MODEL (TRANSFER LEARNING)
# ----------------------------
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(num_classes, activation="softmax")  # 🔥 FIX
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",  # 🔥 FIX
    metrics=["accuracy"]
)

# ----------------------------
# TRAIN
# ----------------------------
model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# ----------------------------
# SAVE MODEL
# ----------------------------
model.save("real_footwear_model.h5")
print("✅ Real footwear model saved!")
