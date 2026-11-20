"""
Program Pengambilan Citra Wajah Anggota Kelompok (3 Mahasiswa)
Menggunakan OpenCV Haar Cascade Classifier
"""

import cv2
import os
import sys
import time

def ambil_foto_webcam(nama_mahasiswa, jumlah_foto=30):
    folder_tujuan = os.path.join("Dataset", nama_mahasiswa)
    os.makedirs(folder_tujuan, exist_ok=True)

    # Inisialisasi Haar Cascade Face Detector
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Kamera/Webcam tidak dapat diakses.")
        print("[INFO] Anda dapat memasukkan foto wajah manual ke dalam folder:", folder_tujuan)
        return False

    print(f"\n[INFO] Mulai pengambilan foto untuk: {nama_mahasiswa}")
    print("[PETUNJUK] Hadapkan wajah ke kamera. Tekan 'SPACE' untuk mengambil foto, atau 'Q' untuk selesai.")

    count = 0
    while count < jumlah_foto:
        ret, frame = cap.read()
        if not ret:
            break

        frame_tampil = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame_tampil, (x, y), (x+w, y+h), (0, 255, 0), 2)

        info_text = f"Mahasiswa: {nama_mahasiswa} | Foto: {count}/{jumlah_foto}"
        cv2.putText(frame_tampil, info_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Pengambilan Dataset Wajah (SPACE: capture, Q: keluar)", frame_tampil)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            if len(faces) > 0:
                x, y, w, h = faces[0]
                face_crop = frame[y:y+h, x:x+w]
                face_resized = cv2.resize(face_crop, (224, 224))
                file_path = os.path.join(folder_tujuan, f"{nama_mahasiswa}_{count+1}.jpg")
                cv2.imwrite(file_path, face_resized)
                count += 1
                print(f"[OK] Tersimpan ({count}/{jumlah_foto}): {file_path}")
            else:
                print("[WARNING] Wajah belum terdeteksi pada frame kamera!")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[SELESAI] Total {count} foto berhasil dikumpulkan untuk {nama_mahasiswa}.")
    return True

def main():
    print("==================================================")
    print("   PENGUMPULAN DATASET WAJAH KELOMPOK (3 ANGGOTA) ")
    print("==================================================")
    mahasiswa_list = ["Mahasiswa_1", "Mahasiswa_2", "Mahasiswa_3"]

    pilihan = input("Gunakan nama default [Mahasiswa_1, 2, 3]? (y/n): ").strip().lower()
    if pilihan != 'y':
        mahasiswa_list = []
        for i in range(1, 4):
            nama = input(f"Masukkan nama mahasiswa {i}: ").strip()
            nama = nama.replace(" ", "_") if nama else f"Mahasiswa_{i}"
            mahasiswa_list.append(nama)

    for nama in mahasiswa_list:
        ambil_foto_webcam(nama, jumlah_foto=20)

if __name__ == "__main__":
    main()
