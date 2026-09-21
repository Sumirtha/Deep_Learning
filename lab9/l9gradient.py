import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(69)

class DeepNetwork(nn.Module):

    def __init__(self, num_layers=50, activation="sigmoid"):
        super(DeepNetwork, self).__init__()

        layers = []

        # Input layer
        layers.append(nn.Linear(10, 10))

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(10, 10))

        self.layers = nn.ModuleList(layers)

        if activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "relu":
            self.activation = nn.ReLU()


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
        return x

def calculate_gradients(activation):

    model = DeepNetwork(num_layers=50,activation=activation)

    # Random input
    x = torch.randn(1, 10)
    # Forward pass
    output = model(x)
    # Simple loss
    loss = output.mean()
    # Backpropagation
    loss.backward()

    gradients = []

    # Collect gradient magnitude from every layer
    for layer in model.layers:
        if layer.weight.grad is not None:
            gradient = layer.weight.grad.abs().mean().item()
            gradients.append(gradient)
    return gradients

# Sigmoid
sigmoid_gradients = calculate_gradients("sigmoid")
relu_gradients = calculate_gradients("relu")

print("\nGradient values for Sigmoid network:")

for i, value in enumerate(sigmoid_gradients):
    print(f"Layer {i+1}: {value:.10f}")

print("\nGradient values for ReLU network:")

for i, value in enumerate(relu_gradients):
    print(f"Layer {i+1}: {value:.10f}")

plt.figure(figsize=(10, 6))

plt.plot(
    range(1, len(sigmoid_gradients) + 1),
    sigmoid_gradients,
    marker='o'
)

plt.yscale("log")

plt.xlabel("Layer")
plt.ylabel("Mean Absolute Gradient")
plt.title("Vanishing Gradient Problem - Sigmoid")

plt.grid(True)

plt.show()

#RelU
plt.figure(figsize=(10, 6))

plt.plot(
    range(1, len(relu_gradients) + 1),
    relu_gradients,
    marker='o'
)

plt.yscale("log")

plt.xlabel("Layer")
plt.ylabel("Mean Absolute Gradient")
plt.title("Gradient Values Across Deep Network - ReLU")

plt.grid(True)

plt.show()


import numpy as np
import matplotlib.pyplot as plt

layers = np.arange(1, 51)

# Vanishing Grdient
vanishing = 0.5 ** layers

# Exploding gradient
exploding = 1.5 ** layers

plt.figure(figsize=(10, 6))

plt.plot(
    layers,
    vanishing,
    marker='o',
    label="Vanishing Gradient"
)

plt.plot(
    layers,
    exploding,
    marker='o',
    label="Exploding Gradient"
)

plt.yscale("log")

plt.xlabel("Number of Layers")
plt.ylabel("Gradient Magnitude")
plt.title("Vanishing and Exploding Gradients")

plt.legend()
plt.grid(True)

plt.show()