import torch 
import torch.nn as nn 

X = torch.tensor(
    [[1.0] , [2.0] , [3.0] , [4.0]]
)

Y = torch.tensor(
    [[3.0], [5.0], [7.0], [9.0]]
)

# Create linear model
#\[\hat y=wx+b\]
model = nn.Linear(1, 1)

# Set initial parameters
with torch.no_grad():
    model.weight.fill_(0.0)
    model.bias.fill_(0.0)

# Loss function
loss_function = nn.MSELoss()

#We update its parameters using gradient descent.

#The formula is:
# \[\boxed{w_{new}=w_{old}-\alpha\frac{\partial L}{\partial w}}\]
 
 # Optimizer
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)
# Training loop
for epoch in range(1000):

    # 1. Forward propagation
    predictions = model(X)

    # 2. Calculate loss
    loss = loss_function(predictions, Y)

    # 3. Clear previous gradients
    optimizer.zero_grad()

    # 4. Backpropagation
    loss.backward()

    # 5. Update parameters
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch {epoch + 1}, "
            f"Loss: {loss.item():.6f}"
        )
# Display learned parameters
print("\nLearned weight:", model.weight.item())
print("Learned bias:", model.bias.item())

# Test prediction
test_input = torch.tensor([[5.0]])

with torch.no_grad():
    prediction = model(test_input)

print("\nPrediction for x=5:", prediction.item())
