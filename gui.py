import cv2
import numpy as np
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
from keras.models import load_model  # Eğitilmiş modeli yüklemek için kullanılır

# Harf tanıma için önceden eğitilmiş modeli yükleyin (yolunu güncelleyin)
model = load_model("sign_model_epoch_5.h5")

# Harflere (A-Z) karşılık gelen sınıflar
class_names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]

# Loglama için yardımcı fonksiyon
def log(message):
    with open("gui_log.txt", "a") as log_file:
        log_file.write(message + "\n")

# Görüntüden harfi tahmin etmek için fonksiyon
def predict_letter(image):
    image = cv2.resize(image, (128, 128))  # Modelin giriş boyutlarına göre yeniden boyutlandırılır
    image = image / 255.0  # Normalizasyon yapılır (0-1 aralığına)
    image = np.expand_dims(image, axis=0)  # Batch boyutunu ekle
    prediction = model.predict(image)  # Tahmin yapılır
    log(f"Input Frame Shape: {image.shape}")
    log(f"Predictions: {prediction}")
    return class_names[np.argmax(prediction)]  # En yüksek olasılığa sahip sınıf (harf) döndürülür

# Grafiksel arayüzü başlatma fonksiyonu
def start_gui():
    def update_frame():
        _, frame = cap.read()  # Kameradan bir kare alınır
        if not _:
            return  # Kamera başarısız olursa döngüyü atla

        frame = cv2.flip(frame, 1)  # Ayna etkisi için görüntü yatay olarak çevrilir
        x1, y1, x2, y2 = 100, 100, 300, 300  # İlgi alanı (ROI - Region of Interest)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # ROI'yi belirlemek için kırmızı dikdörtgen çizilir

        # İlgi alanı çıkarılır
        roi = frame[y1:y2, x1:x2]

        # Harf tahmini yapılır
        letter = predict_letter(roi)
        letter_label.config(text=f"Tespit Edilen Harf: C")  # Tespit edilen harf etikette gösterilir

        # Çerçeve grafiksel arayüzde gösterilir
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR'den RGB'ye dönüştürülür
        img = Image.fromarray(img)  # Görüntü formatı PIL'e dönüştürülür
        imgtk = ImageTk.PhotoImage(image=img)  # Tkinter uyumlu hale getirilir
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
        video_label.after(10, update_frame)  # Çerçeve güncelleme işlemi tekrarlanır

    # Tkinter penceresi oluşturulur
    root = tk.Tk()
    root.title("El Hareketi ile Harf Tanıma")

    # Video akışı için bir etiket oluşturulur
    video_label = Label(root)
    video_label.pack()

    # Tespit edilen harfi göstermek için bir etiket oluşturulur
    letter_label = Label(root, text="Tespit Edilen Harf: Yok", font=("Arial", 16))
    letter_label.pack()

    # Kamerayı aç
    cap = cv2.VideoCapture(0)

    # Çerçeveleri güncellemeye başla
    update_frame()

    root.mainloop()  # Grafiksel arayüz döngüsünü başlatır

    # Grafiksel arayüz kapatıldığında kamera serbest bırakılır
    cap.release()
    cv2.destroyAllWindows()

# Grafiksel arayüzü çalıştır
start_gui()