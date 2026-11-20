import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# LOAD MODEL
model = load_model("model_makanan.keras")

# DAFTAR KELAS (sesuai mapping dataset)
kelas = {
    0: "Bakwan Sayur",
    1: "Bubur Tinotuan",
    2: "Nasi Beras Merah",
    3: "Plencing Kangkung",
    4: "Wortel"
}

def uji_gambar(nama_file):
    if not os.path.exists(nama_file):
        print(f"File {nama_file} tidak ditemukan!")
        return
    print(f"\n================ Pengujian untuk {nama_file} ================")
    # LOAD GAMBAR
    img = image.load_img(
        nama_file,
        target_size=(224, 224)
    )

    # Ubah gambar menjadi array
    img = image.img_to_array(img)

    # Normalisasi
    img = img / 255.0

    # Tambahkan dimensi batch
    img = np.expand_dims(img, axis=0)

    # PREDIKSI
    hasil = model.predict(img)

    # Ambil index dengan nilai terbesar
    index = int(np.argmax(hasil[0]))

    # Ambil nilai probabilitas
    nilai = float(hasil[0][index])

    # HASIL
    print("Nilai        :", nilai)
    print("Index kelas  :", index)
    print("Kelas        :", kelas[index])

if __name__ == "__main__":
    # Pengujian langkah 6 (tes0.jpeg) dan langkah 7 (tes1.jpeg s/d tes4.jpeg)
    for i in range(5):
        uji_gambar(f"tes{i}.jpeg")
