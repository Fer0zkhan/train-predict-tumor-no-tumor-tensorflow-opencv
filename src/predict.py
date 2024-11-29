import cv2
import numpy as np

def predict_image(model, img_path):
    """
    Predict whether an image contains a tumor, automatically detecting the image size.
    """
    # Read the image
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Image at {img_path} could not be read.")

    # Get the original size of the image
    original_size = img.shape[:2]  # (height, width)

    # Resize the image to the model's expected size
    # Assuming the model expects a square input
    model_input_size = model.input_shape[1:3]  # (height, width) of the model input
    img_resized = cv2.resize(img, model_input_size)

    # Normalize the image
    img_resized = img_resized / 255.0  # Scale to [0, 1]. Adjust if needed.

    # Add batch dimension
    img_resized = np.expand_dims(img_resized, axis=0)

    # Predict using the model
    prediction = model.predict(img_resized)[0][0]  # Assuming binary classification with a single output neuron

    # Return prediction
    return {
        "original_size": original_size,
        "prediction": "Tumor Detected" if prediction > 0.5 else "No Tumor Detected"
    }
