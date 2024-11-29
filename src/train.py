from tensorflow.keras.models import load_model

def save_model(model, path="saved_model.h5"):
    model.save(path)

def load_saved_model(path="saved_model.h5"):
    return load_model(path)
