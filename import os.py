import cv2
import os

def record_video(letter, duration, output_path):
    """
    Harf için video kaydı yapar.
    - letter: 'J' veya 'Z'
    - duration: Video kaydı süresi (saniye)
    - output_path: Videonun kaydedileceği yol
    """
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))

    print(f"{letter} harfi için video kaydı başlıyor. Çıkmak için 'Q'ya basın.")
    frame_count = 0
    max_frames = int(duration * 20)  # 20 FPS olduğu varsayılıyor

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            print("Kamera hatası!")
            break
        out.write(frame)
        cv2.imshow('Video Kaydı', frame)
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Video kaydı kullanıcı tarafından durduruldu.")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"{letter} harfi için video kaydı tamamlandı: {output_path}")

def extract_frames(video_path, output_dir, frame_step=5, max_frames=200):
    """
    Videodan kareleri çıkarır ve kaydeder.
    - video_path: Video dosyasının yolu
    - output_dir: Karelerin kaydedileceği klasör
    - frame_step: Kaç karede bir görüntü alınacağı
    - max_frames: Maksimum alınacak kare sayısı
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_frames = 0

    while cap.isOpened() and saved_frames < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_step == 0:  # Her frame_step'te bir kare kaydedilir
            frame_file = os.path.join(output_dir, f'frame_{saved_frames}.jpg')
            cv2.imwrite(frame_file, frame)
            saved_frames += 1
        frame_count += 1

    cap.release()
    print(f"{video_path} videosundan {saved_frames} kare çıkarıldı.")

def augment_images(input_dir):
    """
    Veri artırma işlemi yapar (yansıtma, parlaklık artırma/azaltma).
    - input_dir: İşlenecek görüntülerin bulunduğu klasör
    """
    print("Veri artırma işlemi başlıyor...")
    for file in os.listdir(input_dir):
        img_path = os.path.join(input_dir, file)
        image = cv2.imread(img_path)

        if image is None:
            continue

        # Yatay yansıtma
        flipped_image = cv2.flip(image, 1)
        flipped_path = os.path.join(input_dir, f'flipped_{file}')
        cv2.imwrite(flipped_path, flipped_image)

        # Parlaklık artırma
        bright_image = cv2.convertScaleAbs(image, alpha=1.2, beta=30)
        bright_path = os.path.join(input_dir, f'bright_{file}')
        cv2.imwrite(bright_path, bright_image)

        # Parlaklık azaltma
        dark_image = cv2.convertScaleAbs(image, alpha=0.8, beta=-30)
        dark_path = os.path.join(input_dir, f'dark_{file}')
        cv2.imwrite(dark_path, dark_image)

    print("Veri artırma işlemi tamamlandı.")

# Ana çalışma
if __name__ == "__main__":
    output_directory = './data'  # Veri klasörünün yolu
    duration = 10  # Her video için kayıt süresi (saniye)
    frame_step = 5  # Kare çıkarma aralığı
    datasize = 200  # Toplam elde edilecek kare sayısı

    # J harfi için video ve veri seti oluşturma
    j_video_path = os.path.join(output_directory, 'J_movement.avi')
    j_frames_dir = os.path.join(output_directory, 'J_frames')
    record_video('J', duration, j_video_path)
    extract_frames(j_video_path, j_frames_dir, frame_step, datasize)
    augment_images(j_frames_dir)

    # Z harfi için video ve veri seti oluşturma
    z_video_path = os.path.join(output_directory, 'Z_movement.avi')
    z_frames_dir = os.path.join(output_directory, 'Z_frames')
    record_video('Z', duration, z_video_path)
    extract_frames(z_video_path, z_frames_dir, frame_step, datasize)
    augment_images(z_frames_dir)

    print("Tüm veri toplama ve artırma işlemleri tamamlandı!")
