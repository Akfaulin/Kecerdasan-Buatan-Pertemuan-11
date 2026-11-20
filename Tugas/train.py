"""
Pelatihan Model CNN Klasifikasi Wajah 3 Mahasiswa
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)
BATCH_SIZE = 4
EPOCHS = 15
DATASET_DIR = "Dataset"

def buat_dummy_dataset_wajah_jika_belum_ada():
    """Membuat sampel data wajah jika kamera belum digunakan"""
    if not os.path.exists(DATASET_DIR) or len(os.listdir(DATASET_DIR)) < 3:
        print("[INFO] Membuat sample dataset wajah demonstrasi untuk 3 anggota...")
        from PIL import Image, ImageDraw
        anggota = ["Mahasiswa_1", "Mahasiswa_2", "Mahasiswa_3"]
        colors = [(220, 100, 100), (100, 200, 100), (100, 100, 220)]
        for i, nama in enumerate(anggota):
            folder = os.path.join(DATASET_DIR, nama)
            os.makedirs(folder, exist_ok=True)
            for j in range(1, 11):
                img = Image.new("RGB", (224, 224), color=(240, 240, 240))
                d = ImageDraw.Draw(img)
                # Gambarkan ilustrasi wajah sederhana
                d.ellipse([50, 40, 174, 184], fill=colors[i], outline=(0, 0, 0), width=3)
                d.ellipse([75, 80, 95, 100], fill=(255, 255, 255))
                d.ellipse([129, 80, 149, 100], fill=(255, 255, 255))
                d.rectangle([105, 110, 119, 135], fill=(50, 50, 50))
                d.arc([85, 140, 139, 165], 0, 180, fill=(0, 0, 0), width=3)
                d.text((60, 195), f"{nama} #{j}", fill=(0, 0, 0))
                img.save(os.path.join(folder, f"wajah_{j}.jpg"))

def main():
    buat_dummy_dataset_wajah_jika_belum_ada()

    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        validation_split=0.25
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
    print("\n====================================")
    print("DATASET WAJAH 3 MAHASISWA")
    print("====================================")
    print("Jumlah kelas :", num_classes)
    print("Mapping kelas:", train_data.class_indices)

    model = Sequential([
        Input(shape=(224, 224, 3)),
        Conv2D(32, (3, 3), padding="same", activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), padding="same", activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), padding="same", activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.4),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\n[INFO] Memulai pelatihan model CNN wajah...")
    history = model.fit(train_data, validation_data=val_data, epochs=EPOCHS)

    # Simpan model
    model.save("model_wajah_login.keras")
    np.save("kelas_wajah.npy", train_data.class_indices)
    print("\n[SUKSES] Model tersimpan di 'model_wajah_login.keras'")
    print("[SUKSES] Kelas tersimpan di 'kelas_wajah.npy'")

if __name__ == "__main__":
    main()
