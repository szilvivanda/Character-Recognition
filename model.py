# model.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# Itt már a config.py-ból importáljuk a szükséges konstansokat
from config import IMG_HEIGHT, IMG_WIDTH, CHANNELS, NUM_CLASSES

def create_cnn_model():
    """Létrehozza a Konvolúciós Neurális Hálózat (CNN) modellt."""
    model = keras.Sequential(
        [
            keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS)),

            # A normalizálást (0-255 -> 0-1) a modell végzi
            layers.Rescaling(1. / 255),

            # --- 1. Konvolúciós Blokk (Tervezői Döntés: BatchNormalization) ---
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),

            # --- 2. Konvolúciós Blokk ---
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),

            # --- 3. Konvolúciós Blokk ---
            layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),

            # --- Osztályozó Rész (Tervezői Döntés: Dropout) ---
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),

            # Kimeneti réteg: 62 osztály, 'softmax' aktiváció
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
    )
    return model

def compile_model(model):
    """Lefordítja a modellt a választott optimalizálóval és loss függvénnyel."""
    model.compile(
        # 'sparse_categorical_crossentropy', mert a címkék Keras indexek (0-61)
        loss="sparse_categorical_crossentropy",
        optimizer="adam",  # Tervezői Döntés: ADAM optimalizáló
        metrics=["accuracy"],
    )