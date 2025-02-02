import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, Callback
import os
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Veri setinin bulunduğu klasörün tam yolu
DATASET_PATH = r"C:\Users\CemreDag\Desktop\SignLanguagePython-main\data3"

# Veri setini yükle ve eğitim/test olarak böl
raw_train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    image_size=(128, 128),  # Görüntü boyutları
    batch_size=8,         # Batch boyutu
    validation_split=0.2,  # %20'si doğrulama için
    subset="training",     # Eğitim alt kümesi
    seed=123               # Rastgelelik için seed
)

raw_val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    image_size=(128, 128),  # Görüntü boyutları
    batch_size=8,
    validation_split=0.2,  # %20'si doğrulama için
    subset="validation",   # Doğrulama alt kümesi
    seed=123
)

# Sınıf isimlerini al (ham veri setinden)
class_names = raw_train_dataset.class_names
print("Sınıflar:", class_names)

# Performans için veri önbelleğe al
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = raw_train_dataset.cache().shuffle(100).prefetch(buffer_size=AUTOTUNE)
val_dataset = raw_val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# Veri artırma katmanı
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
])

# Model oluştur
model = models.Sequential([
    data_augmentation,
    layers.Rescaling(1./255, input_shape=(128, 128, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),  # Çıktıyı düzleştirir
    layers.Dense(128, activation='relu'),  # Gizli katman
    layers.Dropout(0.5),  # Dropout katmanı
    layers.Dense(len(class_names), activation='softmax')  # Çıkış katmanı
])

# Modeli derle
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# İlerleme çubuğu için callback oluştur
class TQDMProgressBarCallback(Callback):
    def __init__(self, total_epochs, validation_steps):
        self.total_epochs = total_epochs
        self.validation_steps = validation_steps
        self.epoch_bar = tqdm(total=total_epochs, desc='Epochs', position=0, leave=True)
        self.batch_bar = None

    def on_epoch_begin(self, epoch, logs=None):
        if self.batch_bar:
            self.batch_bar.close()
        self.batch_bar = tqdm(total=self.validation_steps, desc=f'Epoch {epoch+1}/{self.total_epochs}', position=1, leave=False)

    def on_batch_end(self, batch, logs=None):
        self.batch_bar.update(1)

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_bar.update(1)
        self.batch_bar.close()

# Modeli epoch bazlı kaydetmek için callback oluştur
class SaveModelCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        model_name = f"sign_model_epoch_{epoch+1}.h5"
        self.model.save(model_name)
        print(f"Model {model_name} olarak kaydedildi.")

# Modeli eğit ve her epoch sonunda kaydet
total_epochs = 5
validation_steps = len(raw_val_dataset)
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
progress_bar = TQDMProgressBarCallback(total_epochs, validation_steps)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=total_epochs,
    verbose=0,  # Detaylı eğitim çıktısını kapat (tqdm kullanacağımız için)
    callbacks=[SaveModelCallback(), early_stopping, progress_bar]
)
# GPU kullanıldığından emin olun
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))