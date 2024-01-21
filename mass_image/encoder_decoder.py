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

# Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=10, stride=2, padding=1),  # Output size: [8, 128, 128]
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=10, stride=2, padding=1),  # Output size: [8, 64, 64]
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 4, kernel_size=10, stride=2, padding=1)  # Output size: [4, 32, 32]
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 8, kernel_size=10, stride=2, padding=1, output_padding=1),  # Output size: [4, 64, 64]
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 8, kernel_size=10, stride=2, padding=1, output_padding=1),  # Output size: [8, 128, 128]
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=10, stride=2, padding=1, output_padding=1),  # Output size: [8, 256, 256]
            nn.Sigmoid()
        )


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

# Load dataset
dataset = MandelbrotDataset(folder_path='semi_synthetic_photos', transform=transform)
#dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)  # Choose an appropriate batch size
dataloader = DataLoader(dataset, batch_size=600, shuffle=True)  # Choose an appropriate batch size

# Instantiate model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder().to(device)

# Loss and optimizer
criterion = nn.MSELoss()
#optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, amsgrad=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

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
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    if(epoch % 100 == 0):
        # Save the model
        torch.save(model.state_dict(), 'autoencoder.pth')
        print("Model Saved")

# Save the model
torch.save(model.state_dict(), 'autoencoder.pth')
