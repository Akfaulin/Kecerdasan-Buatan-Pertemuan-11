"""
Sistem Login Sederhana dengan Verifikasi Wajah (Face Authentication)
Menggunakan CNN Model & OpenCV
"""

import os
import sys
import cv2
import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = "model_wajah_login.keras"
KELAS_PATH = "kelas_wajah.npy"
CONFIDENCE_THRESHOLD = 0.70  # Ambang batas kepercayaan (70%)

def load_system():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(KELAS_PATH):
        print("[ERROR] Model atau file kelas belum ditemukan. Jalankan train.py terlebih dahulu!")
        return None, None
    model = load_model(MODEL_PATH)
    mapping = np.load(KELAS_PATH, allow_pickle=True).item()
    kelas = {v: k for k, v in mapping.items()}
    return model, kelas

def verifikasi_citra_file(path_gambar, model, kelas):
    if not os.path.exists(path_gambar):
        print(f"[ERROR] File '{path_gambar}' tidak ditemukan.")
        return

    img = cv2.imread(path_gambar)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224)) / 255.0
    img_batch = np.expand_dims(img_resized, axis=0)

    preds = model.predict(img_batch, verbose=0)
    idx = int(np.argmax(preds[0]))
    prob = float(preds[0][idx])
    nama = kelas.get(idx, "Tidak Dikenali")

    print("\n================ HASIL VERIFIKASI ================")
    print(f"File Uji         : {path_gambar}")
    print(f"Identitas Wajah  : {nama}")
    print(f"Confidence       : {prob * 100:.2f}%")

    if prob >= CONFIDENCE_THRESHOLD:
        print("[STATUS] >>> ACCESS GRANTED: Selamat Datang, " + nama + "! <<<")
    else:
        print("[STATUS] >>> ACCESS DENIED: Wajah Tidak Terverifikasi! <<<")

def run_realtime_login(model, kelas):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[WARNING] Webcam tidak terdeteksi. Silakan uji menggunakan gambar file.")
        return

    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    print("\n[INFO] Sistem Login Aktif. Dekatkan wajah ke kamera.")
    print("[INFO] Tekan 'L' untuk melakukan Login Verifikasi. Tekan 'Q' untuk keluar.")

    status_login = "SIAP SCAN"
    warna_status = (255, 255, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(100, 100))

        detected_user = "Mencari Wajah..."
        prob_user = 0.0

        for (x, y, w, h) in faces:
            face_crop = frame[y:y+h, x:x+w]
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (224, 224)) / 255.0
            face_batch = np.expand_dims(face_resized, axis=0)

            preds = model.predict(face_batch, verbose=0)
            idx = int(np.argmax(preds[0]))
            prob_user = float(preds[0][idx])
            detected_user = kelas.get(idx, "Unknown")

            if prob_user >= CONFIDENCE_THRESHOLD:
                box_color = (0, 255, 0)
                label = f"{detected_user} ({prob_user*100:.1f}%)"
            else:
                box_color = (0, 0, 255)
                label = f"TIDAK DIKENAL ({prob_user*100:.1f}%)"

            cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

        # Header UI
        cv2.putText(frame, f"STATUS: {status_login}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, warna_status, 2)
        cv2.putText(frame, "Tekan 'L' untuk Login | 'Q' untuk Keluar", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        cv2.imshow("Sistem Login Verifikasi Wajah CNN", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('l') or key == ord('L'):
            if len(faces) > 0 and prob_user >= CONFIDENCE_THRESHOLD:
                status_login = f"LOGIN BERHASIL: {detected_user}!"
                warna_status = (0, 255, 0)
                print(f"[ACCESS GRANTED] Login Berhasil: {detected_user} ({prob_user*100:.2f}%)")
            else:
                status_login = "LOGIN DITOLAK: Wajah Tidak Valid!"
                warna_status = (0, 0, 255)
                print("[ACCESS DENIED] Wajah tidak cocok atau tingkat keyakinan rendah!")
        elif key == ord('q') or key == ord('Q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    model, kelas = load_system()
    if model is None:
        return

    if len(sys.argv) > 1:
        verifikasi_citra_file(sys.argv[1], model, kelas)
    else:
        print("1. Login Real-time via Webcam")
        print("2. Verifikasi Menggunakan Citra File")
        pilihan = input("Pilih mode (1/2, default: 2): ").strip()
        if pilihan == '1':
            run_realtime_login(model, kelas)
        else:
            # Cari gambar contoh dari dataset untuk diuji
            sample_img = None
            if os.path.exists("Dataset"):
                for root, dirs, files in os.walk("Dataset"):
                    for f in files:
                        if f.endswith(('.jpg', '.png', '.jpeg')):
                            sample_img = os.path.join(root, f)
                            break
                    if sample_img:
                        break
            if sample_img:
                verifikasi_citra_file(sample_img, model, kelas)
            else:
                print("Tidak ditemukan file citra contoh di folder Dataset.")

if __name__ == "__main__":
    main()
