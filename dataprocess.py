# Import necessary libraries
import os
import numpy as np
import cv2
import zipfile
import requests
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Download the dataset
dataset_url = 'https://dicom5c.blob.core.windows.net/public/Data.zip'
response = requests.get(dataset_url)
with open('Data.zip', 'wb') as f:
    f.write(response.content)

# Unzip the dataset
with zipfile.ZipFile('Data.zip', 'r') as zip_ref:
    zip_ref.extractall('Data')

# Dataset paths
images_path = 'Data/Data/TCGA_CS_4941_19960909/images/'
masks_path = 'Data/Data/TCGA_CS_4941_19960909/masks/'

# Load images and masks
image_files = sorted([f for f in os.listdir(images_path) if f.endswith('.tif')])
mask_files = sorted([f for f in os.listdir(masks_path) if f.endswith('.tif')])

# Ensure images and masks are paired
assert len(image_files) == len(mask_files), "Mismatch between images and masks"

# Preprocessing: CLAHE
def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return clahe.apply(img_gray)

# Load images and masks
images = []
masks = []

for img_file, mask_file in zip(image_files, mask_files):
    img_path = os.path.join(images_path, img_file)
    mask_path = os.path.join(masks_path, mask_file)
    
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Load mask as grayscale
    
    if img is not None and mask is not None:  # Check if both image and mask exist
        img = apply_clahe(img)
        images.append(img)
        masks.append(mask)

# Convert to NumPy arrays
images = np.array(images)
masks = np.array(masks)

# Normalize images and masks
images = images / 255.0
masks = masks / 255.0

# Split the dataset
train_images, test_images, train_masks, test_masks = train_test_split(
    images, masks, test_size=0.2, random_state=42
)

print(f"Training samples: {len(train_images)}, Testing samples: {len(test_images)}")
