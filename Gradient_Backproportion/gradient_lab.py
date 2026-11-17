
import torch

# Create a variable that requires gradients
x = torch.tensor(2.0, requires_grad=True) # requires_grad=True	Track operations for differentiation

# Define function
y = x**2 + 3*x + 2

# Calculate derivative
y.backward() # Calculate gradients

print("Input x:", x.item()) # note :- .item():- extracts a python number from a scalar tensor. 
print("Function output:", y.item())
print("Calculated gradient:", x.grad.item()) # .grad :- access calculated gardient



#--------------------------------------------#

# this is the partial derivative 

import torch

x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(1.0, requires_grad=True)

# Define function
f = x**2 + 3*x*y + y**2

# Backpropagation
f.backward()

print("Function output:", f.item())

print("Gradient with respect to x:", x.grad.item())
print("Gradient with respect to y:", y.grad.item())
