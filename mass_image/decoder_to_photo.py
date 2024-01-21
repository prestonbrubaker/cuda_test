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
            nn.Conv2d(64, 30, kernel_size=5, stride=2, padding=2)  # Compressed to 30 feature maps
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(30, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
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

# Create the directory for decoded photos if it doesn't exist
os.makedirs('decoded_photos', exist_ok=True)

# Generate and save 50 decoded images
for i in range(50):
    # Random input for the decoder
    random_input = torch.randn(1, 30, 58, 58)

    # Run the decoder
    with torch.no_grad():
        decoded_image = model.decoder(random_input)

    # Convert to PIL Image
    decoded_image = decoded_image.squeeze(0).squeeze(0).numpy()
    decoded_image = (decoded_image * 255).astype(np.uint8)
    image = Image.fromarray(decoded_image, 'L')
    image = image.resize((256, 256), Image.ANTIALIAS)

    # Save the image
    image.save(f'decoded_photos/decoded_image_{i+1}.png')

print('50 images saved in decoded_photos folder')
