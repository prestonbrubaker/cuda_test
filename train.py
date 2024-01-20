import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Generate dataset
def is_in_mandelbrot(c, max_iter=500):
    z = 0
    for i in range(max_iter):
        z = z*z + c
        if abs(z) > 2:
            return 0  # Not in the Mandelbrot set
    return 1  # In the Mandelbrot set

# Creating a dataset
data_size = 100000  # Number of data points
X = np.random.uniform(-2, 2, (data_size, 2))  # Generate random pairs
y = np.array([is_in_mandelbrot(complex(x[0], x[1])) for x in X])

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Neural network
class MandelbrotNet(nn.Module):
    def __init__(self):
        super(MandelbrotNet, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return torch.sigmoid(self.fc4(x))


# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# Instantiate the model and move it to GPU if available
model = MandelbrotNet().to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Move data to GPU
X = X.to(device)
y = y.to(device)

# Training
epochs = 100000
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs.squeeze(), y)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.6f}')
        with open("loss_values.txt", "a") as file:
            file.write(str(epoch) + " " + str(loss.item()))

# Save the model
model_save_path = "mandelbrot_net.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
