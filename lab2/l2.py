import numpy as np

def relu(z):
    return np.maximum(0, z)

np.random.seed(69)

print("NETWORK 1 FORWARD PASS")

# Input vector x (4 features)
x1 = np.random.randn(4)

# Weight matrix W (1 x 4) and Bias b (1)
W1_net1 = np.random.randn(1, 4)
b1_net1 = np.random.randn(1)

# Linear combination z = Wx + b
z1_net1 = np.dot(W1_net1, x1) + b1_net1

# Activation a = ReLU(z)
a1_net1 = relu(z1_net1)

# Scalar output prediction y_hat
y_hat1 = a1_net1[0]

print(f"Input vector (x): {x1}")
print(f"Pre-activation (z1): {z1_net1}")
print(f"Activation vector (a1): {a1_net1}")
print(f"Final Prediction (y^): {y_hat1:.5f}\n")

print("NETWORK 2 FORWARD PASS")

# Input vector x (4 features)
x2 = np.random.randn(4)

# Layer 1 parameters (4 inputs to 3 neurons)
W1 = np.random.randn(3, 4)
b1 = np.random.randn(3)

# Layer 2 parameters (3 inputs to 2 neurons)
W2 = np.random.randn(2, 3)
b2 = np.random.randn(2)

# Output Layer parameters (2 inputs to 1 neuron)
W3 = np.random.randn(1, 2)
b3 = np.random.randn(1)

# Forward pass Layer 1
z1 = np.dot(W1, x2) + b1
a1 = relu(z1)

# Forward pass Layer 2
z2 = np.dot(W2, a1) + b2
a2 = relu(z2)

# Forward pass Output Layer
z3 = np.dot(W3, a2) + b3
a3 = relu(z3)
y_hat2 = a3[0]

print(f"Input vector (x): {x2}")
print(f"Hidden Layer 1 (z1): {z1}")
print(f"Hidden Layer 1 Activation (a1): {a1}")
print(f"Hidden Layer 2 (z2): {z2}")
print(f"Hidden Layer 2 Activation (a2): {a2}")
print(f"Output Layer (z3): {z3}")
print(f"Output Layer Activation (a3): {a3}")
print(f"Final Prediction (y^): {y_hat2:.5f}")