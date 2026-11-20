import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# LOAD MODEL
model = load_model("model_motor_mobil.keras")

def uji_gambar(nama_file):
    if not os.path.exists(nama_file):
        print(f"File {nama_file} tidak ditemukan!")
        return
    print(f"\n================ Pengujian untuk {nama_file} ================")
    # LOAD GAMBAR
    img = image.load_img(nama_file, target_size=(224, 224))

    # Ubah gambar menjadi array
    img = image.img_to_array(img)

    # Normalisasi
    img = img / 255.0

    # Tambahkan dimensi batch
    img = np.expand_dims(img, axis=0)

    # PREDIKSI
    hasil = model.predict(img)
    nilai = float(hasil[0][0])
    print("Nilai prediksi :", nilai)

    if nilai >= 0.5:
        print("Hasil : MOTOR")
    else:
        print("Hasil : MOBIL")

if __name__ == "__main__":
    # Pengujian langkah 4 untuk tes1.jpeg
    uji_gambar("tes1.jpeg")
    # Pengujian langkah 5 untuk tes2.jpeg
    uji_gambar("tes2.jpeg")
