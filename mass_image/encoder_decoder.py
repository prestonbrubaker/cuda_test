import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import random

# Custom dataset with augmentation
class MandelbrotDataset(Dataset):
    def __init__(self, folder_path, transform=None, augmentation_transform=None, augment_factor=2):
        self.file_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        self.folder_path = folder_path
        self.transform = transform
        self.augmentation_transform = augmentation_transform
        self.augment_factor = augment_factor

    def __len__(self):
        return len(self.file_list) * (1 + self.augment_factor)

    def __getitem__(self, idx):
        file_idx = idx // (1 + self.augment_factor)  # Determine the file index
        img_path = os.path.join(self.folder_path, self.file_list[file_idx])
        image = Image.open(img_path)

        if idx % (1 + self.augment_factor) == 0:
            # Original image
            if self.transform:
                image = self.transform(image)
        else:
            # Augmented image
            if self.augmentation_transform:
                image = self.augmentation_transform(image)

        return image

# Define transformations
original_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
])

augmentation_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.RandomAffine(degrees=30, scale=(0.8, 1.2)), # Rotate and scale
    transforms.ToTensor(),
])

# Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), # Assuming grayscale images
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 5, 7) # Compressed to 5 feature maps
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(5, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid() # Using sigmoid for pixel value range [0, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Load dataset

augment_factor = 2  # Number of augmented images per original image

transform = transforms.Compose([
    transforms.Grayscale(), 
    transforms.ToTensor()
])
dataset = MandelbrotDataset(folder_path='photos', transform=original_transform, 
                            augmentation_transform=augmentation_transform, augment_factor=augment_factor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Instantiate model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder().to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.95)

# Train the model
num_epochs = 1000
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
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the model
torch.save(model.state_dict(), 'autoencoder.pth')
