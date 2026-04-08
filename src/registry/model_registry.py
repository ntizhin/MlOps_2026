import pickle
import os


def save_model(model, path):
    os.makedirs("models", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)