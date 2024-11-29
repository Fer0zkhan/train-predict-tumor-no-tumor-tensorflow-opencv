from tensorflow.keras.utils import Sequence
import cv2
import os
import numpy as np
import pandas as pd

class DataGenerator(Sequence):
    """
    Custom data generator to load images in batches.
    """
    def __init__(self, images_path, labels_path, img_size, batch_size):
        self.images_path = images_path
        self.batch_size = batch_size
        self.img_size = img_size
        
        # Read the CSV file
        self.df = pd.read_csv(labels_path)
        self.image_paths = self.df['filepath'].values
        self.labels = (self.df['tumor_mask_empty'] == 0).astype(int).values
        
        self.indexes = np.arange(len(self.image_paths))
    
    def __len__(self):
        # Number of batches per epoch
        return int(np.ceil(len(self.image_paths) / self.batch_size))
    
    def __getitem__(self, index):
        # Get batch indexes
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_images = []
        batch_labels = []
        
        # Load images for this batch
        for i in batch_indexes:
            img_path = os.path.join(self.images_path, os.path.basename(self.image_paths[i]))
            if os.path.exists(img_path):
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (self.img_size, self.img_size)) / 255.0
                batch_images.append(img)
                batch_labels.append(self.labels[i])
        
        return np.array(batch_images, dtype=np.float32), np.array(batch_labels, dtype=np.int32)
    
    def on_epoch_end(self):
        # Shuffle indexes after each epoch
        np.random.shuffle(self.indexes)
