import torch
import torch.nn as nn
import pygame
import numpy as np

# Neural network definition (should match the training architecture)
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

# Pygame setup
pygame.init()
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Mandelbrot Set Visualization")

# Function to convert screen coordinates to complex numbers
def screen_to_complex(x, y, width, height, scale=4):
    return (x - width / 2) / (width / scale), (y - height / 2) / (height / scale)

# Visualization loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Iterate over each pixel on the screen
    for x in range(width):
        for y in range(height):
            re, im = screen_to_complex(x, y, width, height)
            input_tensor = torch.tensor([[re, im]], dtype=torch.float32)
            prediction = model(input_tensor).item()

            # Color based on the model's prediction
            color = (0, 255, 0) if prediction > 0.5 else (0, 0, 255)
            screen.set_at((x, y), color)

    pygame.display.flip()

pygame.quit()
