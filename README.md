# Privacy-Enhanced-Gait-Recognition

## Project Overview:
The primary objective of the project is to develop a gait recognition system that ensures the privacy of individuals captured through video surveillance. Gait recognition is a biometric technology that identifies individuals based on their walking patterns. Unlike traditional methods that store images in a
recognizable format, this project employs encoding techniques to transform the images into an unreadable format. The encoded data is stored securely, ensuring that no individuals can access and interpret the information.

## Key Features
Our project gait recognition, prioritizes user privacy in a climate of growing surveillance anxieties. Our approach invloves transforming surveillance by storing data as encoded images, which enhances privacy by making the data unreadable in the event of unauthorized access. This method aligns with the increasing emphasis on ethical use of biometric data, providing a model for secure, private identification practices in advanced surveillance. The workflow established in the `app.py` script closely mirrors real-world scenarios. Users submit encrypted images, along with the corresponding encryption key and initialization vector (IV). The
system dynamically decrypts the images using the provided key and IV, creating a seamless integration with the model trained on decrypted images. Further enhancing privacy, we encrypt the encoded representations before comparison, adding an extra layer of protection against potential breaches. By fusing Siamese networks and CNNs with encryption, our project sets a new standard for secure and privacy-preserving gait recognition.
