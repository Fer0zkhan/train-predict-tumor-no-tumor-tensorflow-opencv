# Liver Tumor Detection Using Deep Learning: A Segmentation and Classification Approach


## **Overview**
This project focuses on the **automatic detection of liver tumors** in CT images using a combination of **deep learning classification** and **image segmentation techniques**. The goal is to assist radiologists in accurately diagnosing liver tumors by providing both a **prediction** and a **visualization** of the tumor's location.

---

## **Features**
1. **Single Image Prediction**:
   - Predict whether a single image contains a tumor (`Tumor Detected` or `No Tumor Detected`).
   - Optionally visualize tumor regions using segmentation masks.

2. **Bulk Image Prediction**:
   - Predict tumor presence for multiple images in a directory or from a CSV file.
   - Save predictions and tumor visualizations.

3. **Visualization**:
   - Draw bounding boxes around detected tumor regions based on segmentation masks.
   - Save processed images with bounding boxes for review.

4. **Output**:
   - Save predictions and processed images in the specified output directory.

---

## **Technologies Used**
- **Deep Learning**: TensorFlow/Keras for model training and prediction.
- **Image Processing**: OpenCV for preprocessing, contour detection, and visualization.
- **Data Handling**: NumPy and Pandas for efficient data manipulation.
- **Dataset**: [LiTS Dataset](https://www.kaggle.com/datasets/andrewmvd/lits-png) for liver and tumor segmentation.

---

## **Setup and Installation**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Fer0zkhan/train-predict-tumor-no-tumor-tensorflow-opencv.git
   cd train-predict-tumor-no-tumor-tensorflow-opencv
   ```

2. **Install Dependencies**:
   Create a virtual environment and install the required libraries:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Download Dataset**:
   - Use the [LiTS Dataset](https://www.kaggle.com/datasets/andrewmvd/lits-png).
   - Place the dataset inside the `dataset_6/` directory or update the paths in the script accordingly.

---

## **How to Use**

### **First You have to train the model using the dateset_6**
```bash
python main.py
``` 

### If you want to use pre-trained data then don't run 
```bash
python main.py
``` 

### **After Train the Model**

### 1. **Single Image Prediction**
To predict tumor presence for a single image:
```bash
python predict_tumor.py -s /path/to/image.png
```

### 2. **Single Image with Mask**
To predict tumor presence and visualize the tumor using a segmentation mask:
```bash
python predict_tumor.py -s /path/to/image.png -m /path/to/mask.png
```

### 3. **Bulk Image Prediction**
To predict tumor presence for all images in a directory:
```bash
python predict_tumor.py -b /path/to/images/
```

### 4. **Bulk Prediction with CSV**
To predict using a CSV file:
```bash
python predict_tumor.py -b /path/to/images/ -t /path/to/csv_file.csv
```
The CSV file should have the following columns:
- `filepath`: Path to the image.
- `maskpath` (optional): Path to the segmentation mask.

---

## **Project Structure**
```plaintext
.
├── dataset_6/                    # Dataset directory
│   ├── dataset/                  # CT images
│   ├── lits_df.csv
│   ├── lits_probe.csv
│   ├── lits_test.csv
│   ├── lits_train.csv
├── src/                          # Source code
│   ├── data_loader.py            # Handles data loading and preprocessing
│   ├── model.py                  # Defines the CNN model
│   ├── train.py                  # Model training script
│   ├── evaluate.py               # Evaluation and metrics
│   ├── predict.py                # Prediction logic
│   ├── utils.py                  # Utility functions (e.g., save/load model)
├── predict_tumor.py              # Main script for prediction and visualization
├── main.py                       # Script for train the model
├── .gitignore                    # GitIgnore
├── requirements.txt              # Python dependencies
├── saved_model.h5                # Pre Trained Model
└── README.md                     # Project documentation
```

---

## **Implementation Details**

### **Model**
- A **Convolutional Neural Network (CNN)** is used for binary classification:
  - **Input**: CT images resized to the required dimensions (e.g., 128x128).
  - **Output**: Probability of tumor presence.
- The model is trained using binary cross-entropy loss and optimized with the Adam optimizer.

### **Workflow**
1. **Preprocessing**:
   - CT images are resized and normalized to [0, 1].
   - Masks are resized to match the image dimensions for accurate visualization.
2. **Prediction**:
   - The CNN model predicts tumor presence based on the image.
   - A threshold (`0.5`) is used to classify the image as `Tumor Detected` or `No Tumor Detected`.
3. **Visualization**:
   - Contours of the tumor are extracted from the mask using OpenCV.
   - Bounding boxes are drawn around detected tumor regions.

---

## **Results**

- **Single Image Output**:
  - Prediction: `Tumor Detected` or `No Tumor Detected`.
  - Visualization: Processed image with bounding boxes saved in `predicted_images_with_boxes/`.

- **Bulk Prediction Output**:
  - A dictionary of predictions for each image.
  - Processed images with bounding boxes saved.

---

## **Applications**
- **Medical Diagnostics**: Assist radiologists in liver tumor detection.
- **Research**: Automate tumor segmentation and classification in medical imaging.
- **Education**: A learning resource for deep learning and computer vision techniques in healthcare.

---

## **Future Improvements**
- Implement a fully automated segmentation model to generate masks dynamically.
- Improve robustness by using larger datasets for training and validation.
- Add support for other imaging modalities (e.g., MRI, PET scans).

---

## **License**
This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

## **Acknowledgments**
- [LiTS Dataset](https://www.kaggle.com/datasets/andrewmvd/lits-png)
- OpenCV, TensorFlow/Keras, and other open-source libraries.

---
