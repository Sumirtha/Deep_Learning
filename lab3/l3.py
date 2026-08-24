import numpy as np

# Derivative of ReLU activation function
def relu_derivative(z):
    return (z > 0).astype(float)

def relu(z):
    return np.maximum(0, z)

np.random.seed(69)
y_true = 1.0  # loss: L = 0.5 * (y_hat - y)^2

# NETWORK 1 BACKWARD PASS (4 inputs to 1 output)
# Forward pass setup
x1 = np.random.randn(4)
W1_net1 = np.random.randn(1, 4)
b1_net1 = np.random.randn(1)

z1_net1 = np.dot(W1_net1, x1) + b1_net1
a1_net1 = relu(z1_net1)
y_hat1 = a1_net1[0]

# Backward pass
dL_da1 = y_hat1 - y_true                         # Loss gradient wrt output: dL/d(y_hat)
dz1_net1 = dL_da1 * relu_derivative(z1_net1)     # dL/dz1 = (dL/da1) * relu'(z1)
dW1_net1 = np.outer(dz1_net1, x1)                # dL/dW1 = dz1 * x^T
db1_net1 = dz1_net1                              # dL/db1 = dz1

print("NETWORK 1 GRADIENTS")
print("Neuron Pre-activation Gradient (dz1):", dz1_net1)
print("Weight Gradients (dW1):\n", dW1_net1)
print("Bias Gradient (db1):", db1_net1)

# NETWORK 2 BACKWARD PASS (4 to 3 to 2 to 1)
# Forward pass setup
x2 = np.random.randn(4)
W1 = np.random.randn(3, 4); b1 = np.random.randn(3)
W2 = np.random.randn(2, 3); b2 = np.random.randn(2)
W3 = np.random.randn(1, 2); b3 = np.random.randn(1)

z1 = np.dot(W1, x2) + b1; a1 = relu(z1)
z2 = np.dot(W2, a1) + b2; a2 = relu(z2)
z3 = np.dot(W3, a2) + b3; a3 = relu(z3)
y_hat2 = a3[0]

# Output Layer (Layer 3)
dL_da3 = y_hat2 - y_true
dz3 = dL_da3 * relu_derivative(z3)
dW3 = np.outer(dz3, a2)
db3 = dz3

# Hidden Layer 2
da2 = np.dot(W3.T, dz3)
dz2 = da2 * relu_derivative(z2)
dW2 = np.outer(dz2, a1)
db2 = dz2

# Hidden Layer 1
da1 = np.dot(W2.T, dz2)
dz1 = da1 * relu_derivative(z1)
dW1 = np.outer(dz1, x2)
db1 = dz1

print("\nNETWORK 2 GRADIENTS ")
print("Layer 3 Neuron Gradient (dz3):", dz3)
print("Layer 3 Weight Gradient (dW3):\n", dW3)
print("Layer 2 Neuron Gradient (dz2):", dz2)
print("Layer 2 Weight Gradient (dW2):\n", dW2)
print("Layer 1 Neuron Gradient (dz1):", dz1)
print("Layer 1 Weight Gradient (dW1):\n", dW1)