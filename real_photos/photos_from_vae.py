import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# Global variable for the dimension of the latent space
LATENT_DIM = 12


# Variational Autoencoder model
class VariationalAutoencoder(nn.Module):
    def __init__(self):
        super(VariationalAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 24, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU()
        )
        # Adjust the input features of the following layers based on the encoder output
        self.fc_mu = nn.Linear(in_features=24*64*64, out_features=LATENT_DIM)
        self.fc_log_var = nn.Linear(in_features=24*64*64, out_features=LATENT_DIM)

        # Decoder
        # Adjust the output features to match the input of the first transposed conv layer
        self.decoder_input = nn.Linear(in_features=LATENT_DIM, out_features=24*64*64)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(24, 24, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.ConvTranspose2d(24, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, 24, 64, 64)
        x = self.decoder(x)
        return x

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var


# Load the trained model
model = VariationalAutoencoder()
model.load_state_dict(torch.load('variational_autoencoder.pth'))
model.eval()

# Create the directory for generated photos if it doesn't exist
os.makedirs('decoded_photos', exist_ok=True)

# Generate and save images
num_images = 50  # Number of images to generate
for i in range(num_images):
    # Generate random latent vectors
    random_latent_vector = torch.randn(1, LATENT_DIM)


    # Decode the random latent vectors
    with torch.no_grad():
        generated_image = model.decode(random_latent_vector).cpu()

    # Convert to PIL Image and save
    generated_image = generated_image.squeeze().numpy()  # Remove batch and channel dimensions
    generated_image = (generated_image * 255).astype(np.uint8)  # Convert to 8-bit pixel values
    image = Image.fromarray(generated_image, 'L')
    image = image.resize((256, 256), Image.ANTIALIAS)
    image.save(f'decoded_photos/decoded_vae_image_{i+1}.png')

print(f'{num_images} images saved in decoded_photos folder')
