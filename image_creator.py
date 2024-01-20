import torch
import torch.nn as nn
from PIL import Image
import numpy as np

# Neural network definition
class MandelbrotNet(nn.Module):
    def __init__(self):
        super(MandelbrotNet, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))

# Load the trained model
model = MandelbrotNet()
model.load_state_dict(torch.load("mandelbrot_net.pth"))
model.eval()

# Image setup
width, height = 800, 600
image = Image.new("RGB", (width, height))

# Function to convert image coordinates to complex numbers
def image_to_complex(x, y, width, height, scale=4):
    return (x - width / 2) / (width / scale), (y - height / 2) / (height / scale)

# Iterate over each pixel in the image
for x in range(width):
    for y in range(height):
        re, im = image_to_complex(x, y, width, height)
        input_tensor = torch.tensor([[re, im]], dtype=torch.float32)
        prediction = model(input_tensor).item()

        # Color based on the model's prediction
        color = (0, 255, 0) if prediction > 0.5 else (0, 0, 255)
        image.putpixel((x, y), color)

# Save the image
image.save("mandelbrot_set.png")
