import os
import cv2
import pandas as pd
from src.utils import load_saved_model
from src.predict import predict_image
from tqdm import tqdm

IMG_SIZE = 128
OUTPUT_DIR = "predicted_images_with_boxes"

def draw_bounding_box(image_path, mask_path, prediction):
    """
    Draw a bounding box on the image based on the tumor mask and save the modified image.
    """
    # Load the image and mask
    img = cv2.imread(image_path)
    if img is None:
        print(f"Unable to read image: {image_path}")
        return
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Unable to read mask: {mask_path}")
        return

    # Ensure the mask size matches the image size
    if mask.shape[:2] != img.shape[:2]:
        print(f"Resizing mask from {mask.shape[:2]} to match image size {img.shape[:2]}")
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

    # Apply binary threshold to highlight the tumor region
    _, thresholded_mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

    # Save the thresholded mask for debugging
    # cv2.imwrite("debug_thresholded_mask.png", thresholded_mask)

    # Find contours in the thresholded mask
    contours, _ = cv2.findContours(thresholded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Contours found: {len(contours)}")

    # Minimum contour area to filter out noise
    MIN_CONTOUR_AREA = 50
    for contour in contours:
        if cv2.contourArea(contour) < MIN_CONTOUR_AREA:  # Skip small contours
            continue
        x, y, w, h = cv2.boundingRect(contour)
        color = (0, 0, 255) if prediction['prediction'] == "Tumor Detected" else (0, 255, 0)
        thickness = 2
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)

    # Save the image to the output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    output_path = os.path.join(OUTPUT_DIR, os.path.basename(image_path))
    cv2.imwrite(output_path, img)
    print(f"Saved: {output_path}")

def predict_single_image(model, image_path):
    """
    Predict tumor presence for a single image.
    """
    result = predict_image(model, image_path)
    print(f"Prediction for {image_path}: {result}")
    return result

def predict_single_with_mask(model, image_path, mask_path):
    """
    Predict tumor presence for a single image and draw bounding box using the mask.
    """
    result = predict_single_image(model, image_path)
    draw_bounding_box(image_path, mask_path, result)

def predict_bulk_images(model, images_dir):
    """
    Predict tumor presence for all images in a directory.
    """
    predictions = {}
    images = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for image_path in tqdm(images, desc="Processing images"):
        result = predict_single_image(model, image_path)
        predictions[os.path.basename(image_path)] = result

    print("\nBulk Predictions:")
    for img, res in predictions.items():
        print(f"{img}: {res}")
    return predictions

def predict_from_csv(model, csv_path):
    """
    Predict tumor presence using a CSV file with file paths and masks.
    """
    print(f"Loading data from CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    if "filepath" not in df.columns:
        print("Error: CSV must contain a 'filepath' column.")
        return

    predictions = []
    for _, row in tqdm(df.iterrows(), desc="Processing CSV", total=len(df)):
        image_path = row['filepath']
        mask_path = row.get('maskpath', None)

        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        if mask_path and os.path.exists(mask_path):
            predict_single_with_mask(model, image_path, mask_path)
        else:
            result = predict_single_image(model, image_path)
            predictions.append((image_path, result))

    print("\nCSV Predictions Complete.")

def main():
    import argparse

    # Argument parser setup
    parser = argparse.ArgumentParser(description="Predict tumor presence in images.")
    parser.add_argument("-s", "--single", type=str, help="Path to a single image.")
    parser.add_argument("-m", "--mask", type=str, help="Path to a mask for the single image.")
    parser.add_argument("-b", "--bulk", type=str, help="Path to a directory containing images.")
    parser.add_argument("-t", "--csv", type=str, help="Path to a CSV file for bulk prediction.")

    args = parser.parse_args()

    # Load the saved model
    model = load_saved_model()

    # Single image prediction
    if args.single:
        if args.mask:
            predict_single_with_mask(model, args.single, args.mask)
        else:
            predict_single_image(model, args.single)

    # Bulk prediction
    elif args.bulk:
        if args.csv:
            predict_from_csv(model, args.csv)
        else:
            predict_bulk_images(model, args.bulk)

    else:
        print("Please provide valid arguments. Use -h for help.")

if __name__ == "__main__":
    main()
