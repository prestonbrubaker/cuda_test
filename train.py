import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Generate dataset
def is_in_mandelbrot(c, max_iter=100):
    z = 0
    for i in range(max_iter):
        z = z*z + c
        if abs(z) > 2:
            return 0  # Not in the Mandelbrot set
    return 1  # In the Mandelbrot set

# Creating a dataset
data_size = 1000  # Number of data points
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
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))

# Instantiate the model
model = MandelbrotNet()

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs.squeeze(), y)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Save the model
model_save_path = "mandelbrot_net.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# Testing with a new point
test_point = torch.tensor([[0.3, 0.5]], dtype=torch.float32)
prediction = model(test_point).item()
print(f'Prediction for {test_point}: {"In" if prediction > 0.5 else "Out"} of the Mandelbrot set')
