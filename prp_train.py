import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


# Define the source and destination paths
src_path = "C:/Users/abhay/OneDrive/Desktop/Lens_classification/dataset/train"
dest_path = "C:/Users/abhay/OneDrive/Desktop/Lens_classification/dataset_preprocessed/train"

# Define the image transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
    transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomRotation(degrees=30),
    transforms.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0), ratio=(1.0, 1.0))
])

# Loop through each class and preprocess the images
for class_name in os.listdir(src_path):
    class_dir = os.path.join(src_path, class_name)
    for filename in os.listdir(class_dir):
        # Load the image from the .npy file
        npy_path = os.path.join(class_dir, filename)
        image_array = np.load(npy_path)

        # Transpose the tensor to (height, width, channels  )
        image_array = np.transpose(image_array, (1, 2, 0))

        # Apply the transform to the image array
        image_tensor = transform(image_array)

        # Save the image as a .jpg file
        jpg_path = os.path.join(dest_path, class_name, filename[:-4] + ".jpg")
        os.makedirs(os.path.dirname(jpg_path), exist_ok=True)
        Image.fromarray((image_tensor.squeeze().numpy() * 0.5 + 0.5) * 255).convert('RGB').save(jpg_path)
