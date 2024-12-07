import os
import cv2
import string

# Veri klasörünün yolu
DATA_DIR = './datamodelegitim'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# İngilizce alfabe harfleri (J ve Z hariç)
alphabet = [letter for letter in string.ascii_uppercase if letter not in ['J', 'Z']]
dataset_size = 40  # Her harf için alınacak görüntü sayısı

# Kamerayı başlat
cap = cv2.VideoCapture(0)  # Kamera indeksini kontrol et, gerekiyorsa 0 yerine 1 veya 2 dene.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Genişlik
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Yükseklik

# Aynalı görüntü düzeltmesi için flip ayarı
flip = True

for letter in alphabet:
    # Her harf için klasör oluştur
    letter_dir = os.path.join(DATA_DIR, letter)
    if not os.path.exists(letter_dir):
        os.makedirs(letter_dir)

    # Mevcut verileri kontrol et
    existing_files = len(os.listdir(letter_dir)) if os.path.exists(letter_dir) else 0
    if existing_files >= dataset_size:
        print(f'Data for letter {letter} is already collected.')
        continue  # Eğer veri zaten tamamlanmışsa, bir sonraki harfe geç

    print(f'Collecting data for letter {letter} (already collected: {existing_files})')
    # Başlamadan önce kullanıcıdan hazır olmasını iste
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kamera kare yakalayamıyor!")
            break

        # Aynalı görüntüyü düzelt
        if flip:
            frame = cv2.flip(frame, 1)

        # Kullanıcıya hazır olduğunu bildiren metin ekle
        cv2.putText(frame, f'Ready to capture {letter}? Press "Q" to start.', 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Görüntüleri toplama
    counter = existing_files + 1
    while counter <= dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Kamera kare yakalayamıyor!")
            break

        # Aynalı görüntüyü düzelt
        if flip:
            frame = cv2.flip(frame, 1)

        # Görüntüyü gri tonlamaya çevir ve boyutlandır
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized_frame = cv2.resize(gray_frame, (160, 120))  # Daha küçük boyutlar

        # Görüntüyü kaydet
        file_name = os.path.join(letter_dir, f'{letter}{counter}.jpg')
        cv2.imwrite(file_name, resized_frame)

        # Görüntüyü göster
        cv2.imshow('frame', resized_frame)

        counter += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"Data collection for {letter} interrupted.")
            break

cap.release()
cv2.destroyAllWindows()
print("Veri toplama tamamlandı!")
