import os
from PIL import Image
import torch
from torchvision import transforms

def preprocess_and_save_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    # Count total number of .png files in the input folder
    total_files = sum(1 for f in os.listdir(input_folder) if f.endswith('.png'))
    processed_count = 0

    for filename in os.listdir(input_folder):
        if filename.endswith('.png'):
            file_path = os.path.join(input_folder, filename)
            image = Image.open(file_path)
            tensor_image = transform(image)
            torch.save(tensor_image, os.path.join(output_folder, filename + '.pt'))

            # Update and display progress
            processed_count += 1
            progress_percent = (processed_count / total_files) * 100
            print(f"Processed {processed_count}/{total_files} files ({progress_percent:.2f}%)")

input_folder = 'semi_synthetic_photos'
output_folder = 'tensor_database'
preprocess_and_save_images(input_folder, output_folder)
