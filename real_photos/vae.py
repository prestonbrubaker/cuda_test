import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import random

# Model Parameters
latent_dim = 100  # Example latent space dimension

# Instantiate the model
model = VariationalAutoencoder(latent_dim).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.file_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        self.folder_path = folder_path
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.file_list[idx])
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        if self.transform:
            image = self.transform(image)
        return image


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(VariationalAutoencoder, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1), # 64x48
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # 32x24
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 16x12
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # 8x6
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(256 * 8 * 6, latent_dim)
        self.fc_log_var = nn.Linear(256 * 8 * 6, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 256 * 8 * 6)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
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
        x = x.view(-1, 256, 8, 6)
        return self.decoder(x)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var


def loss_function(recon_x, x, mu, log_var):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD


# Load dataset

# Define transformation
transform = transforms.Compose([
    transforms.Resize((128, 96)),  # Resize to a smaller size
    transforms.Grayscale(), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Assuming grayscale images
])

# Global variable to choose which dataset to use
USE_PREPROCESSED_DATASET = False  # Set to True to use preprocessed dataset

# Then, in your main script where you load the dataset:
if USE_PREPROCESSED_DATASET:
    dataset = MandelbrotDataset(folder_path='photos', transform=transform)

else:
    dataset = MandelbrotDataset(folder_path='photos', transform=transform)

dataloader = DataLoader(dataset, batch_size=60, shuffle=True)


# Instantiate VAE model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VariationalAutoencoder().to(device)

# Loss and optimizer
# For VAE, use the custom loss function that includes both BCE and KLD
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.00, amsgrad=False)
#optimizer = optim.SGD(model.parameters(), lr=0.00000001, momentum=0.9)




# Train the model
num_epochs = 100000
for epoch in range(num_epochs):
    for data in dataloader:
        img = data.to(device)

        # Forward pass
        recon_batch, mu, log_var = model(img)

        # Calculate loss
        loss = loss_function(recon_batch, img, mu, log_var)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')
    if(epoch % 25 == 0):
        # Save the model
        torch.save(model.state_dict(), 'variational_autoencoder.pth')
        print("Model Saved")


# Save the model
torch.save(model.state_dict(), 'variational_autoencoder.pth')

