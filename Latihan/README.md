# Panduan Pengerjaan Latihan Modul 11 (Plant Leaves Classification)

Dataset yang digunakan bersumber dari Kaggle:
[Plant Leaves for Image Classification](https://www.kaggle.com/datasets/csafrit2/plant-leaves-for-image-classification)

## Cara Menggunakan:
1. Unduh dataset dari tautan di atas (atau subset Plants_2).
2. Ekstrak folder dataset citra ke dalam folder `Latihan/Dataset/` sehingga strukturnya menjadi:
   ```
   Latihan/
     Dataset/
       <Nama_Tanaman_1>/
         gambar1.jpg
         gambar2.jpg
       <Nama_Tanaman_2>/
         ...
     train.py
     test.py
   ```
3. Jalankan pelatihan model:
   ```bash
   python train.py
   ```
4. Jalankan pengujian prediksi:
   ```bash
   python test.py
   # atau untuk gambar spesifik:
   python test.py path/ke/gambar.jpg
   ```
5. File model yang tersimpan adalah `model_plant_leaves.keras` beserta grafik pelatihan `grafik_latihan_plant.png`.
