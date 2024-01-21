import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# Define the autoencoder model (same as before)
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 7)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(3, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
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
    random_input = torch.randn(1, 3, 58, 58)

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
