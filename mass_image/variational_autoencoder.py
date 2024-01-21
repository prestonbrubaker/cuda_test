import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import random

# Global variable for the dimension of the latent space
LATENT_DIM = 7


# Custom Dataset
class MandelbrotDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.file_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        self.folder_path = folder_path
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.file_list[idx])
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image

class TensorDataset(Dataset):
    def __init__(self, folder_path):
        self.file_list = [f for f in os.listdir(folder_path) if f.endswith('.pt')]
        self.folder_path = folder_path

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        tensor_path = os.path.join(self.folder_path, self.file_list[idx])
        tensor = torch.load(tensor_path)
        return tensor


# Variational Autoencoder model
class VariationalAutoencoder(nn.Module):
    def __init__(self):
        super(VariationalAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        # Adjust the input features of the following layers based on the encoder output
        self.fc_mu = nn.Linear(in_features=8*64*64, out_features=LATENT_DIM)
        self.fc_log_var = nn.Linear(in_features=8*64*64, out_features=LATENT_DIM)

        # Decoder
        # Adjust the output features to match the input of the first transposed conv layer
        self.decoder_input = nn.Linear(in_features=LATENT_DIM, out_features=8*64*64)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
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
        x = x.view(-1, 8, 64, 64)
        x = self.decoder(x)
        return x

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

# Loss function
def loss_function(recon_x, x, mu, log_var):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD


# Load dataset

# Define transformation
transform = transforms.Compose([
    transforms.Grayscale(), 
    transforms.ToTensor()
])


# Global variable to choose which dataset to use
USE_PREPROCESSED_DATASET = True  # Set to True to use preprocessed dataset

# Then, in your main script where you load the dataset:
if USE_PREPROCESSED_DATASET:
    dataset = TensorDataset(folder_path='tensor_database')
else:
    dataset = MandelbrotDataset(folder_path='semi_synthetic_photos', transform=transform)

dataloader = DataLoader(dataset, batch_size=600, shuffle=True)


# Instantiate VAE model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VariationalAutoencoder().to(device)

# Loss and optimizer
# For VAE, use the custom loss function that includes both BCE and KLD
optimizer = optim.Adam(model.parameters(), lr=0.005, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, amsgrad=True)

# Train the model
num_epochs = 100000
for epoch in range(num_epochs):
    for data in dataloader:
        img = data.to(device)
        # Forward pass
        recon_batch, mu, log_var = model(img)

        
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
