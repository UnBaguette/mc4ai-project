#Libraries
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2

# Load Dataset
def load_dataset(data_path):
    images = []
    labels = []
    label_map = {chr(i + ord('A')): i for i in range(26)}  # A-Z mapped to 0-25

    for folder_name in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder_name)
        if os.path.isdir(folder_path):
            label = label_map[folder_name]  # Assign label based on folder name
            for filename in os.listdir(folder_path):
                img_path = os.path.join(folder_path, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    # Resize the image to 64x64
                    img = cv2.resize(img, (64, 64))
                    # Convert the image to grayscale
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    images.append(np.array(img))
                    labels.append(label)

    return np.array(images), np.array(labels)
