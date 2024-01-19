import torch
import numpy as np 

data = [[1, 2], [3, 4]]     # Create a normal array of data

x_data = torch.tensor(data)   # Transform the array into a tensor

print(str(x_data)) 

x_ones = torch.ones_like(x_data)    # Transform the data into a tensor of ones

x_rand = torch.rand_like(x_data, dtype=torch.float)     # Transform the array into a tensor of random floats

print(x_ones)
print(x_rand)

tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")   # Display the width and height of tensor as an array
print(f"Datatype of tensor: {tensor.dtype}")    # Display the type of data in the tensor such as float or interger
print(f"Device tensor is stored on: {tensor.device}")   # Display if the tensor is stored on the cpu or gpu

if torch.cuda.is_available():
    tensor = tensor.to("cuda")  # Reassign the tensor moved to CUDA

print(f"Device tensor is stored on: {tensor.device}")   # Display if the tensor is stored on the cpu or gpu

print(torch.cuda.is_available())  # Should return True if CUDA is properly installed
print(torch.cuda.get_device_name(0))  # Should return the name of your CUDA GPU

print(f"Device tensor is stored on: {tensor.device}")   # Display if the tensor is stored on the cpu or gpu
