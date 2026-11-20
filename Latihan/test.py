"""
Program Pengujian Model Klasifikasi Citra Daun Tanaman
"""

import os
import sys
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

def main():
    model_path = "model_plant_leaves.keras"
    if not os.path.exists(model_path):
        print(f"[ERROR] File model '{model_path}' tidak ditemukan. Jalankan train.py terlebih dahulu.")
        return

    # Load mapping kelas jika tersedia
    if os.path.exists("class_indices.npy"):
        class_indices = np.load("class_indices.npy", allow_pickle=True).item()
        kelas = {v: k for k, v in class_indices.items()}
    else:
        kelas = {0: "Kelas 0", 1: "Kelas 1", 2: "Kelas 2"}

    model = load_model(model_path)
    print("[INFO] Model berhasil dimuat.")
    print("Daftar kelas terdaftar:", kelas)

    # Uji citra yang diberikan atau citra default dari folder dataset
    test_target = sys.argv[1] if len(sys.argv) > 1 else None

    if test_target and os.path.exists(test_target):
        targets = [test_target]
    else:
        # Cari beberapa gambar contoh dari Dataset jika ada
        targets = []
        if os.path.exists("Dataset"):
            for root, dirs, files in os.walk("Dataset"):
                for f in files:
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        targets.append(os.path.join(root, f))
                        if len(targets) >= 3:
                            break
                if len(targets) >= 3:
                    break

    if not targets:
        print("[WARNING] Tidak ada file citra untuk diuji. Gunakan argumen path file: python test.py <path_gambar>")
        return

    print("\n================ HASIL PREDIKSI CITRA DAUN ================")
    for img_path in targets:
        img = image.load_img(img_path, target_size=(224, 224))
        arr = image.img_to_array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)

        preds = model.predict(arr, verbose=0)
        idx = int(np.argmax(preds[0]))
        prob = float(preds[0][idx])

        print(f"\nFile Citra    : {img_path}")
        print(f"Probabilitas  : {prob * 100:.2f}%")
        print(f"Index Kelas   : {idx}")
        print(f"Nama Spesies  : {kelas.get(idx, 'Unknown')}")

if __name__ == "__main__":
    main()
