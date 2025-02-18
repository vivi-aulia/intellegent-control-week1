import cv2
import face_recognition
from datetime import datetime


def detect_age_from_face(image_path):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    if len(face_locations) == 0:
        return "Wajah tidak terdeteksi. Coba lagi dengan gambar yang lebih jelas."

    # Simulasi deteksi umur (biasanya menggunakan model AI untuk estimasi umur)
    estimated_age = 25  # Gantilah dengan model prediksi umur yang sesuai
    return estimated_age


# Menggunakan kamera untuk mengambil gambar
video_capture = cv2.VideoCapture(0)
ret, frame = video_capture.read()
cv2.imwrite("captured_face.jpg", frame)
video_capture.release()

# Mendeteksi umur dari gambar wajah
age = detect_age_from_face("captured_face.jpg")
print(f"Estimasi umur Anda adalah {age} tahun.")