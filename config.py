# config.py
# --- Útvonalak ---
TRAIN_DIR = "all_train_data"
TEST_DIR = "all_test_data"

# --- Kép Konfiguráció ---
IMG_HEIGHT = 32
IMG_WIDTH = 32
CHANNELS = 1  # Szürkeárnyalatos
NUM_CLASSES = 62  # 0-9, A-Z, a-z

# --- Tanítási Hiperparaméterek ---
EPOCHS = 20
BATCH_SIZE = 64
SEED = 1337
VALIDATION_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 3