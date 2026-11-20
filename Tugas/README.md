# Panduan Tugas: Sistem Login dengan Verifikasi Wajah (Face Authentication CNN)

Tugas ini mengimplementasikan sistem login verifikasi wajah berbasis Convolutional Neural Network (CNN) untuk 3 anggota kelompok, sesuai capaian pada Bab 11 halaman 25.

## Tahapan Pengerjaan:
1. **Pengambilan Dataset Wajah Anggota Kelompok (3 Mahasiswa)**:
   Jalankan script:
   ```bash
   python capture_dataset.py
   ```
   - Masukkan nama masing-masing anggota.
   - Wajah akan dideteksi secara otomatis menggunakan OpenCV Haar Cascade Classifier.
   - Tekan tombol **SPASI** untuk mengambil foto wajah (disarankan 20-30 foto per anggota dengan berbagai ekspresi dan pencahayaan).
   - Tekan **Q** jika sudah selesai untuk anggota tersebut.

2. **Pelatihan Model CNN Wajah**:
   Jalankan script:
   ```bash
   python train.py
   ```
   - Skrip akan membaca dataset dari folder `Dataset/`, melakukan data augmentasi, melatih model CNN 3 kelas, dan menyimpan model ke `model_wajah_login.keras` serta mapping kelas ke `kelas_wajah.npy`.

3. **Pengujian & Sistem Login Verifikasi**:
   Jalankan script:
   ```bash
   # Mode interaktif (webcam / file):
   python login_system.py

   # Atau langsung verifikasi file gambar:
   python login_system.py path/ke/foto_wajah.jpg
   ```
   - Jika wajah sesuai dengan salah satu anggota kelompok dengan tingkat kepercayaan (confidence) >= 70%, sistem akan menampilkan:
     `ACCESS GRANTED: Selamat Datang, [Nama Anggota]!`
   - Jika bukan anggota kelompok atau confidence rendah, sistem akan menampilkan:
     `ACCESS DENIED: Wajah Tidak Dikenali!`
