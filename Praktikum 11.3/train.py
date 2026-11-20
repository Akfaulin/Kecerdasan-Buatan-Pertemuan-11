import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==========================
# DATASET
# ==========================
datagen = ImageDataGenerator(
    rescale=1./255
)

train_data = datagen.flow_from_directory(
    "Dataset",
    target_size=(224, 224),
    batch_size=2,
    class_mode="categorical"
)

print("Jumlah data :", train_data.samples)
print("Kelas :", train_data.class_indices)

# ==========================
# CNN
# ==========================
model = Sequential([
    Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), padding="same", activation="relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(5, activation="softmax")
])

# ==========================
# COMPILE
# ==========================
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ==========================
# TRAINING
# ==========================
model.fit(
    train_data,
    epochs=10
)

# ==========================
# SAVE
# ==========================
model.save("model_makanan.keras")
print("Model berhasil disimpan sebagai model_makanan.keras")
