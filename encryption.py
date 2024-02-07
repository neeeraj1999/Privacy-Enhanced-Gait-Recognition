import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

# Function to pad message to be a multiple of 16 bytes
def pad(message):
    while len(message) % 16 != 0:
        message += b'\x00'
    return message

# Function to encrypt an image
def encrypt_image(image_path, encrypted_path, key, iv):
    with open(image_path, "rb") as image_file:
        raw_image = image_file.read()
    padded_image = pad(raw_image)
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    encrypted_image = encryptor.update(padded_image) + encryptor.finalize()
    with open(encrypted_path, "wb") as file:
        file.write(encrypted_image)

# Paths
casia_dataset_path = "GaitDatasetA-silh"  # Update this path
encrypted_dataset_path = "Encrypted"  # Update this path

# Encryption Key and IV
key = os.urandom(16)  # Ensure to keep this key safe for decryption
iv = os.urandom(16)   # Same for the IV

# Encrypting the CASIA A dataset
for root, dirs, files in os.walk(casia_dataset_path):
    for file in files:
        # Original and new file paths
        original_file_path = os.path.join(root, file)
        relative_path = os.path.relpath(original_file_path, casia_dataset_path)
        encrypted_file_path = os.path.join(encrypted_dataset_path, relative_path)

        # Creating directories if they don't exist
        os.makedirs(os.path.dirname(encrypted_file_path), exist_ok=True)

        # Encrypting and saving the file
        encrypt_image(original_file_path, encrypted_file_path, key, iv)


print(key)
print(iv)