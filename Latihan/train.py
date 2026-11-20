"""
Program Pelatihan Model CNN untuk Klasifikasi Citra Daun Tanaman (Plant Leaves)
Dataset referensi: https://www.kaggle.com/datasets/csafrit2/plant-leaves-for-image-classification
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# PARAMETER
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15
DATASET_DIR = "Dataset"

def buat_dummy_dataset_jika_belum_ada():
    """Membuat contoh struktur dataset jika pengguna belum mengunduh dari Kaggle"""
    if not os.path.exists(DATASET_DIR) or len(os.listdir(DATASET_DIR)) == 0:
        print("[INFO] Folder Dataset belum ditemukan atau kosong.")
        print("[INFO] Membuat sample dataset untuk demonstrasi arsitektur CNN...")
        from PIL import Image, ImageDraw
        classes = ["Alstonia_Scholaris", "Arjun", "Gauva", "Jamun", "Jatropha"]
        for c in classes:
            folder = os.path.join(DATASET_DIR, c)
            os.makedirs(folder, exist_ok=True)
            for i in range(1, 6):
                filepath = os.path.join(folder, f"{c.lower()}_{i}.jpg")
                if not os.path.exists(filepath):
                    img = Image.new("RGB", (224, 224), color=(34, 139, 34))
                    d = ImageDraw.Draw(img)
                    d.ellipse([40, 30, 184, 194], fill=(46, 204, 113), outline=(20, 90, 50), width=3)
                    d.line([(112, 30), (112, 194)], fill=(20, 90, 50), width=2)
                    d.text((50, 100), f"{c}\nSample {i}", fill=(255, 255, 255))
                    img.save(filepath)
        print("[INFO] Sample dataset berhasil dibuat. Anda dapat menimpa dengan dataset asli dari Kaggle kapan saja.")

def main():
    buat_dummy_dataset_jika_belum_ada()

    # Preprocessing & Data Augmentation
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_data = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True
    )

    val_data = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False
    )

    num_classes = len(train_data.class_indices)
    print("====================================")
    print("INFORMASI DATASET TANAMAN")
    print("====================================")
    print("Jumlah kelas    :", num_classes)
    print("Daftar kelas    :", train_data.class_indices)
    print("Sampel training :", train_data.samples)
    print("Sampel validasi :", val_data.samples)

    # Arsitektur Model CNN Multi-Kelas
    model = Sequential([
        Input(shape=(224, 224, 3)),
        Conv2D(32, (3, 3), padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    # Pelatihan Model
    print("\n[INFO] Memulai training model CNN...")
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS
    )

    # Simpan visualisasi metrik evaluasi
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    if "val_accuracy" in history.history:
        plt.plot(history.history["val_accuracy"], label="Val Accuracy")
    plt.title("Akurasi Model")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("Loss Model")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("grafik_latihan_plant.png")
    print("[INFO] Grafik training disimpan ke grafik_latihan_plant.png")
    plt.close()

    # Simpan Model
    model_path = "model_plant_leaves.keras"
    model.save(model_path)
    print(f"[INFO] Model berhasil disimpan ke {model_path}")

    # Simpan mapping kelas
    np.save("class_indices.npy", train_data.class_indices)
    print("[INFO] Mapping kelas disimpan ke class_indices.npy")

if __name__ == "__main__":
    main()
