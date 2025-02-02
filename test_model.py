import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2

# Eğitilmiş modeli yükle
model = load_model("sign_model_epoch_5.h5")  # 5. epoch sonunda kaydedilen model
class_names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]

# Loglama için yardımcı fonksiyon
def log(message):
    with open("test_log.txt", "a") as log_file:
        log_file.write(message + "\n")

# Varsayılan kamerayı açmaya çalış
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # DirectShow kullan

if not cap.isOpened():
    print("Kamera açılmadı.")
    log("Kamera açılmadı.")
    exit()  # Kamera açılamazsa çıkış yap

print("Kamera başarıyla açıldı.")
log("Kamera başarıyla açıldı.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Görüntü alınamıyor.")
        log("Görüntü alınamıyor.")
        break

    # Aynalı görüntüyü düzelt
    frame = cv2.flip(frame, 1)

    # Model için görüntüyü yeniden boyutlandır
    resized_frame = cv2.resize(frame, (128, 128))  # 128x128 boyutunda olacak
    normalized_frame = resized_frame / 255.0  # Görüntüleri normalize et
    input_frame = np.expand_dims(normalized_frame, axis=0)  # Batch boyutunu ekle

    # Tahmin yap ve logla
    predictions = model.predict(input_frame)
    log(f"Input Frame Shape: {input_frame.shape}")
    log(f"Predictions: {predictions}")
    predicted_class = class_names[np.argmax(predictions)]  # En yüksek tahmini al
    log(f"Predicted Class: {predicted_class}")

    # Tahmini ekrana yazdır
    cv2.putText(frame, f'Prediction: {predicted_class}',
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Sign Language Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' tuşuna basınca çıkılır
        break

# Kamera açıldıktan sonra döngü tamamlandıktan sonra serbest bırakılır.
cap.release()  # Kamerayı serbest bırak
cv2.destroyAllWindows()  # Tüm pencereleri kapat