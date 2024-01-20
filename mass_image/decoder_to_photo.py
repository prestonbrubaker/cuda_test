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
            nn.Conv2d(32, 5, 7)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(5, 32, 7),
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

# Random input for the decoder
# Assuming the compressed size is 64 feature maps of 64x64 (as in the defined model)
random_input = torch.randn(1, 5, 5, 5)  # Batch size, Channels, Height, Width

# Run the decoder
with torch.no_grad():
    decoded_image = model.decoder(random_input)

# Convert to PIL Image and save
decoded_image = decoded_image.squeeze(0).squeeze(0).numpy()  # Remove batch and channel dimensions
decoded_image = (decoded_image * 255).astype(np.uint8)  # Convert to 8-bit pixel values
image = Image.fromarray(decoded_image, 'L')
image = image.resize((256, 256), Image.ANTIALIAS)
image.save('decoded_image.png')

print('Image saved as decoded_image.png')
