import torch
import torch.nn as nn
from PIL import Image
import numpy as np

# Neural network definition
class MandelbrotNet(nn.Module):
    def __init__(self):
    def __init__(self):
        super(MandelbrotNet, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 10)
        self.fc6 = nn.Linear(10, 10)
        self.fc7 = nn.Linear(10, 10)
        self.fc8 = nn.Linear(10, 10)
        self.fc9 = nn.Linear(10, 10)
        self.fc10 = nn.Linear(10, 10)
        self.fc11 = nn.Linear(10, 10)
        self.fc12 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = torch.relu(self.fc7(x))
        x = torch.relu(self.fc8(x))
        x = torch.relu(self.fc9(x))
        x = torch.relu(self.fc10(x))
        x = torch.relu(self.fc11(x))
        return torch.sigmoid(self.fc12(x))

# Load the trained model
model = MandelbrotNet()
model.load_state_dict(torch.load("mandelbrot_net.pth"))
model.eval()

# Image setup
width, height = 1600, 1200    #800, 600
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
        if(prediction < 0):
            color = (0, 0, 0)
        elif(prediction >= 0 and prediction <= 1):
            color = (int(255 * prediction) ,int(255 * prediction),int(255 * prediction))
        else:
            color = (255, 255, 255)
        image.putpixel((x, y), color)

# Save the image
image.save("mandelbrot_set.png")
