# main.py
# Importáljuk a moduljainkat
from config import EPOCHS, EARLY_STOPPING_PATIENCE
from model import create_cnn_model, compile_model
from data_pipeline import load_train_validation_datasets, load_test_data
# További szükséges importok
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

# A 4. lépésből átvett karakter-leképezés
ALL_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
INT_TO_CHAR = {i: char for i, char in enumerate(ALL_CHARS)}


def run_project():
    print("TensorFlow Verzió:", tf.__version__)
    print("--- Adatbetöltés és Pipeline Készítése... ---")

    # 4. LÉPÉS: Adatbetöltő Pipeline Hívása
    try:
        train_ds, val_ds = load_train_validation_datasets()
    except FileNotFoundError as e:
        print(e)
        return

    print("--- CNN Modell Felépítése és Fordítása... ---")

    # 5. és 6. LÉPÉS: Modell Felépítése és Fordítása
    model = create_cnn_model()
    model.summary()
    compile_model(model)

    # 7. LÉPÉS: Modell Tanítása
    print("\n--- Modell Tanítása... ---")
    early_stopper = keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True
    )

    # Tanítás
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=[early_stopper]
    )

    print("\n--- Tanítás Kész! ---")

    # 8. LÉPÉS: Kiértékelés (Opcionális - ide kerül a plot hívása, ha szükséges)
    # ... Pl. plt.plot(history.history['val_accuracy']) ...

    # Tanulási görbék rajzolása
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Tanítási Pontosság')
    plt.plot(history.history['val_accuracy'], label='Validációs Pontosság')
    plt.title('Pontosság')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Tanítási Veszteség')
    plt.plot(history.history['val_loss'], label='Validációs Veszteség')
    plt.title('Veszteség (Loss)')
    plt.legend()
    plt.show()

    # 9. LÉPÉS: Predikció és Beadandó Fájl Létrehozása
    print("\n--- Predikció a Teszt Adatokon... ---")


    # Teszt adatok betöltése
    X_test, test_image_ids = load_test_data()
    if X_test.size == 0:
        print("Nincsenek teszt képek a predikcióhoz.")
        return

    # Predikció
    predictions_proba = model.predict(X_test)
    predicted_indices = np.argmax(predictions_proba, axis=1) # Keras index: 0-61

    # KONVERZIÓ JAVÍTÁSA: Keras index -> Karakter (0-9, a-z, A-Z)
    # A feladat a karakterazonosító (Character ID) előrejelzését kéri.
    # EHELYETT:predicted_chars = [INT_TO_CHAR[idx] for idx in predicted_indices]
    #EZ:
    predicted_labels = predicted_indices + 1

    # Beadandó DataFrame
    submission_df = pd.DataFrame({
    'class': predicted_labels, # Első oszlop
    'TestImage': test_image_ids # Második oszlop
    })

    submission_filename = "OCRpredictions.txt"
    submission_df.to_csv(
        submission_filename,
        sep=';',
        index=False,
        header=True
    )

    print(f"\n--- Kész! A beadandó fájl '{submission_filename}' néven elmentve. ---")
    print(f"A fájl itt található: {os.path.abspath(submission_filename)}")

    submission_filename = "OCRpredictions.csv"
    submission_df.to_csv(
        submission_filename,
        sep=';',  # <--- PONTOSVESSZŐ HASZNÁLATA
        index=False,
        header=True  # A fejléc megtartása
    )

    print(f"\n--- Kész! A beadandó fájl '{submission_filename}' néven elmentve. ---")
    print(f"A fájl itt található: {os.path.abspath(submission_filename)}")


if __name__ == "__main__":
    run_project()