from tensorflow.keras.models import load_model

def save_model(model, path="saved_model.h5"):
    model.save(path)

def load_saved_model(path="saved_model.h5"):
    model = load_model(path, compile=False)  # Avoid compilation warning
    model.compile(metrics=["accuracy"])  # Add a dummy metric
    return model
