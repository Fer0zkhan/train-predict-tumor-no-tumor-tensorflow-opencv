from src.data_loader import DataGenerator
from src.model import build_model
from src.utils import save_model

# Define paths
IMAGES_PATH = "./dataset_6/dataset_6/"
LABELS_PATH = "./dataset_6/lits_train.csv"
IMG_SIZE = 128
EPOCHS = 20
BATCH_SIZE = 32

# Create data generator for training
train_generator = DataGenerator(IMAGES_PATH, LABELS_PATH, IMG_SIZE, BATCH_SIZE)

# Build and train model
model = build_model(IMG_SIZE)
model.fit(train_generator, epochs=EPOCHS)

# Save the trained model
save_model(model)
