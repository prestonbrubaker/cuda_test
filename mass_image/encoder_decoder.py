import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import random

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


# Autoencoder model
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

        # Apply the weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

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


# Instantiate model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder().to(device)

# Loss and optimizer
criterion = nn.MSELoss()
#optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, amsgrad=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

# Train the model
num_epochs = 100000
for epoch in range(num_epochs):
    for data in dataloader:
        img = data.to(device)
        # Forward pass
        output = model(img)
        loss = criterion(output, img)
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')
    if(epoch % 100 == 0):
        # Save the model
        torch.save(model.state_dict(), 'autoencoder.pth')
        print("Model Saved")

# Save the model
torch.save(model.state_dict(), 'autoencoder.pth')
