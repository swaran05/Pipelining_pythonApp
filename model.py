from tensorflow.keras.models import load_model
import os

MODEL_PATH = "model.h5"

def get_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. Please run train_once.py locally and commit model.h5."
        )
    return load_model(MODEL_PATH)
