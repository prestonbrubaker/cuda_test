import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# Define the autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder
        # Each Conv2d layer halves the spatial dimensions
        # Input size: [1, 256, 256], Output size after layer: [5, 128, 128]
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=3, stride=2, padding=1),  
            nn.BatchNorm2d(5),
            nn.ReLU(),
            # Input size: [5, 128, 128], Output size after layer: [5, 64, 64]
            nn.Conv2d(5, 5, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(5),
            nn.ReLU(),
            # Input size: [5, 64, 64], Output size after layer: [5, 32, 32]
            nn.Conv2d(5, 2, kernel_size=3, stride=2, padding=1)
        )
        
        # Decoder
        # Each ConvTranspose2d layer doubles the spatial dimensions
        # Input size: [5, 32, 32], Output size after layer: [5, 64, 64]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2, 5, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(5),
            nn.ReLU(),
            # Input size: [5, 64, 64], Output size after layer: [5, 128, 128]
            nn.ConvTranspose2d(5, 5, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(5),
            nn.ReLU(),
            # Input size: [5, 128, 128], Output size after layer: [5, 256, 256]
            nn.ConvTranspose2d(5, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Load the trained model
model = Autoencoder()
model.load_state_dict(torch.load('autoencoder.pth'))
model.eval()

# Create the directory for generated photos if it doesn't exist
os.makedirs('decoded_photos', exist_ok=True)

# Generate and save images
num_images = 50  # Number of images to generate
for i in range(num_images):
    # Random input for the decoder
    # The dimensions here should match the output dimensions of your encoder
    random_input = torch.randn(1, 4, 256, 256)  # Example dimensions

    # Run the decoder
    with torch.no_grad():
        generated_image = model.decoder(random_input)

    # Convert to PIL Image and save
    generated_image = generated_image.squeeze().numpy()  # Remove batch and channel dimensions
    generated_image = (generated_image * 255).astype(np.uint8)  # Convert to 8-bit pixel values
    image = Image.fromarray(generated_image, 'L')
    image = image.resize((256, 256), Image.ANTIALIAS)
    image.save(f'decoded_photos/decoded_image_{i+1}.png')

print(f'{num_images} images saved in decoded_photos folder')
