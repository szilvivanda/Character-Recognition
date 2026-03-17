# data_pipeline.py
import tensorflow as tf
import os
import glob
import numpy as np
import cv2
from config import TRAIN_DIR, TEST_DIR, IMG_HEIGHT, IMG_WIDTH, CHANNELS, BATCH_SIZE, SEED, VALIDATION_SPLIT

def load_train_validation_datasets():
    """Betölti a tanító és validációs adathalmazokat a Keras utility-val."""
    if not os.path.exists(TRAIN_DIR):
        raise FileNotFoundError(f"Hiba: A '{TRAIN_DIR}' mappa nem található!")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=SEED,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        color_mode='grayscale',
        batch_size=BATCH_SIZE
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        color_mode='grayscale',
        batch_size=BATCH_SIZE
    )

    # Optimalizáljuk a betöltést a gyorsabb tanításért
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds

def load_test_data():
    """Betölti a teszt képeket, előkészíti a predikcióra, és visszaadja a fájlazonosítókat."""
    if not os.path.exists(TEST_DIR):
        raise FileNotFoundError(f"Hiba: A '{TEST_DIR}' mappa nem található!")

    test_image_paths = glob.glob(f"{TEST_DIR}/**/*.png", recursive=True)
    test_image_paths.sort()

    test_images_list = []
    test_image_ids = []

    for img_path in test_image_paths:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue

        img_resized = cv2.resize(img, (IMG_WIDTH, IMG_WIDTH))
        img_final = np.reshape(img_resized, (IMG_HEIGHT, IMG_WIDTH, CHANNELS))

        test_images_list.append(img_final)
        test_image_ids.append(os.path.basename(img_path))

    X_test = np.array(test_images_list)
    return X_test, test_image_ids