import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# Model Parameters
latent_dim = 5  # Example latent space dimension
LATENT_DIM = latent_dim

class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(VariationalAutoencoder, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1), # 320x240
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=1), # 160x120
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=1), # 80x60
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=1), # 40x30
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(16 * 40 * 30, latent_dim)
        self.fc_log_var = nn.Linear(16 * 40 * 30, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 16 * 40 * 30)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1), # 80x60
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1), # 160x120
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1), # 320x240
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1), # 640x480
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
        x = x.view(-1, 16, 40, 30)
        return self.decoder(x)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

def loss_function(recon_x, x, mu, log_var):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD


# Load the trained model
model = VariationalAutoencoder(LATENT_DIM)
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

    # Resize image to 640x480 before saving
    image = image.resize((640, 480), Image.ANTIALIAS)
    image.save(f'decoded_photos/decoded_vae_image_{i+1}.png')

print(f'{num_images} images saved in decoded_photos folder')
