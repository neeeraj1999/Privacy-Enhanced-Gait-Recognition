import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda, UpSampling2D
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

# Function to load and preprocess images in a directory
def load_and_preprocess_images_in_sequence(directory_path, sequence_length=10, target_size=(128, 128)):
    sequences = []
    labels = []

    classes = sorted(os.listdir(directory_path))

    for class_index, class_name in enumerate(classes):
        class_path = os.path.join(directory_path, class_name)

        for sequence_folder in os.listdir(class_path):
            sequence_path = os.path.join(class_path, sequence_folder)

            image_sequence = []

            for file_name in os.listdir(sequence_path):
                file_path = os.path.join(sequence_path, file_name)

                # Load and preprocess the image
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
                image = cv2.resize(image, target_size)
                image = image / 255.0  # Normalize to [0, 1]

                # Append the preprocessed image to the sequence
                image_sequence.append(image)

            # Make sure the sequence has the desired length
            if len(image_sequence) >= sequence_length:
                # Concatenate the images into a single vector
                sequence_vector = np.concatenate(image_sequence[:sequence_length], axis=-1)

                # Append the sequence vector and its corresponding label
                sequences.append(sequence_vector)
                labels.append(class_index)

    # Convert lists to NumPy arrays
    sequences = np.array(sequences)
    labels = np.array(labels)

    return sequences, labels

# Load and preprocess data
directory_path = "/content/GaitDatasetA-silh"
sequence_length = 10
target_size = (128, 128)
X, y = load_and_preprocess_images_in_sequence(directory_path, sequence_length=sequence_length, target_size=target_size)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)