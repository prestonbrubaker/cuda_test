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
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 5, kernel_size=5, stride=2, padding=2)  # Compressed to 5 feature maps
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(5, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=5, stride=2, padding=2, output_padding=1),
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
    random_input = torch.randn(1, 5, 16, 16)  # Example dimensions

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
