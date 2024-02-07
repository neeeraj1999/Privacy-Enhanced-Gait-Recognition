import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from PIL import Image
import io
import os
from collections import Counter


# Define the CNN model
class GaitRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(GaitRecognitionModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128 * 56 * 56, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# Function to pad message to be a multiple of 16 bytes
def pad(message):
    while len(message) % 16 != 0:
        message += b'\x00'
    return message

# Function to decrypt an image
def decrypt_image(encrypted_image, key, iv):
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    decrypted_image = decryptor.update(encrypted_image) + decryptor.finalize()
    return decrypted_image

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []

        for image_file in sorted(os.listdir(root_dir)):
            image_path = os.path.join(root_dir, image_file)
            if os.path.isfile(image_path):
                self.image_paths.append(image_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image_path = self.image_paths[idx]
            with open(image_path, "rb") as encrypted_file:
                encrypted_image = encrypted_file.read()

            return encrypted_image, image_path
        except Exception as e:
            print(f"Error processing file {image_path}: {e}")
            raise e
        
from PIL import Image
import numpy as np



# Main function for decryption and identification
def decrypt_and_identify(filepath, key, iv, model, transform):
    try:
        # Initialize the dataset with the specified folder
        custom_dataset = CustomDataset(root_dir=filepath, transform=None)

        # Initialize results list to store identities
        results = []

        for encrypted_image_data, image_path in custom_dataset:
            # Decrypt the image
            decrypted_image_data = decrypt_image(encrypted_image_data, key, iv)
            print(f"Decrypted image {image_path}")
            # Convert the decrypted data to a NumPy array
            image_array = np.frombuffer(decrypted_image_data, dtype=np.uint8)

            # Create a PIL image from the NumPy array
            image = Image.fromarray(image_array).convert("RGB")

            # Apply the transformation
            if transform:
                image = transform(image)

            # Convert the image to a tensor and add a batch dimension
            image = image.unsqueeze(0)

            # Pass the image through the model
            with torch.no_grad():
                output = model(image)
            

            # Identify the person
            identity = identify_person(output)


            # Store the result
            results.append(identity)

        ridentity = np.argmax(np.bincount(results))
        return ridentity

    except Exception as e:
        return results


def identify_person(output):
    _, predicted_class = torch.max(output, 1)
    
    if predicted_class.item() == 0:
        return 0
    elif predicted_class.item() == 1:
        return 1
    else:
        return f"Unknown Person with Predicted Class: {predicted_class.item()}"



# Example usage
if __name__ == "__main__":
    # Input filepath, key, and iv
    encrypted_dataset_path = "Test/00_1"
    key = b'\xb7\xc6Jv\xcef\xf2z\xeai;\xe5\x1a\xd0z\\'  # Replace with your key
    iv = b'\x00>~\xcb\xd047\xa0\xe8\x94\xfea\xf0\xa4\x051'  # Replace with your IV


    # Load the saved Gait Recognition model
    gait_model = GaitRecognitionModel(num_classes=2)  # Adjust the number of classes accordingly
    gait_model.load_state_dict(torch.load('gait_recognition_model.pth'))
    gait_model.eval()

    # Define the transformation for the model
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Decrypt and identify the images using the pre-trained models
    results = decrypt_and_identify(encrypted_dataset_path, key, iv, gait_model, transform)

    if results == 0:
      print("The person identity is X")
    else:
      print("The person identity is Y")    
