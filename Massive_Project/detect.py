import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
from pathlib import Path
import pathlib
import pygame
import time
from collections import deque

# Mengubah Path agar kompatibel dengan Windows
pathlib.PosixPath = pathlib.WindowsPath

# Path ke model YOLOv5 yang sudah dilatih
model_name = './yolov5/runs/exp/weights/best (2).pt'

# Memuat model YOLOv5
model = torch.hub.load('./yolov5', 'custom', path=model_name, source='local', force_reload=True)

# Inisialisasi Pygame untuk memutar alarm
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound('alarm.wav')  # Ganti dengan path file alarm Anda

# Menggunakan kamera video (0 menunjukkan kamera default)
cap = cv2.VideoCapture(0)

# Variabel untuk menyimpan prediksi dalam periode waktu tertentu
prediction_counts = {
    'DangerousDriving': deque(maxlen=6),
    'Distracted': deque(maxlen=6),
    'Drinking': deque(maxlen=6),
    'SleepyDriving': deque(maxlen=6),
    'Yawn': deque(maxlen=6)
}

# Jangka waktu deteksi dalam detik
time_limits = {
    'DangerousDriving': 2,
    'Distracted': 10,
    'Drinking': 4,
    'SleepyDriving': 5,
    'Yawn': 6
}

# Fungsi untuk memutar alarm dengan durasi 5 detik
def play_alarm():
    if not pygame.mixer.get_busy():  # Cek jika tidak ada suara yang sedang diputar
        alarm_sound.play()
        time.sleep(3)  # Tunggu selama 5 detik
        alarm_sound.stop()  # Hentikan alarm

# Loop untuk membaca setiap frame dari kamera
while cap.isOpened():
    # Membaca frame dari kamera
    ret, frame = cap.read()
    
    if not ret:
        break

    # Mengonversi frame ke grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Mengonversi kembali grayscale ke format 3 channel
    gray_frame_3_channel = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
    
    # Mendapatkan hasil deteksi objek dari model YOLOv5
    results = model(gray_frame_3_channel)

    # Memeriksa hasil deteksi
    labels = results.pandas().xyxy[0]['name'].tolist()
    current_time = time.time()

    # Update counts dan kondisi untuk setiap kelas
    for label in prediction_counts.keys():
        if label in labels:
            prediction_counts[label].append(current_time)
            if len(prediction_counts[label]) >= 6 and (current_time - prediction_counts[label][0] <= time_limits[label]):
                play_alarm()
        else:
            # Hapus waktu yang sudah terlalu lama
            while prediction_counts[label] and (current_time - prediction_counts[label][0] > time_limits[label]):
                prediction_counts[label].popleft()

    # Menampilkan frame dengan hasil deteksi
    cv2.imshow('YOLO', np.squeeze(results.render()))

    # Menunggu tombol 'q' untuk keluar dari loop
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Melepaskan kamera dan menutup jendela tampilan
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()