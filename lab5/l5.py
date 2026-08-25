import numpy as np
import matplotlib.pyplot as plt

# XOR Dataset Definition
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([[0], [1], [1], [0]])

# Activation Function & Derivative
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

#  Network Architecture & Weight Initialization
np.random.seed(69)

input_dim = 2  # x1, x2
hidden_dim = 4  # 4 hidden neurons
output_dim = 1  # XOR output prediction

# Initialize weights randomly between -1 and 1, biases to zero
W1 = np.random.uniform(-1, 1, (input_dim, hidden_dim))
b1 = np.zeros((1, hidden_dim))

W2 = np.random.uniform(-1, 1, (hidden_dim, output_dim))
b2 = np.zeros((1, output_dim))

learning_rate = 1.0
epochs = 5000
loss_history = []

# Training Loop (Forward + Backward Pass)
for epoch in range(epochs):
    # Forward Pass
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)

    # Loss Computation
    loss = np.mean(0.5 * (a2 - y) ** 2)
    loss_history.append(loss)

    # Backward Pass
    # Layer 2 Gradients (Output Layer)
    dz2 = (a2 - y) * sigmoid_derivative(a2)
    dW2 = np.dot(a1.T, dz2)
    db2 = np.sum(dz2, axis=0, keepdims=True)

    # Layer 1 Gradients (Hidden Layer)
    da1 = np.dot(dz2, W2.T)
    dz1 = da1 * sigmoid_derivative(a1)
    dW1 = np.dot(X.T, dz1)
    db1 = np.sum(dz1, axis=0, keepdims=True)

    # Gradient Descent Updates
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

# Output
print("XOR PREDICTIONS")
for i in range(len(X)):
    print(f"Input: {X[i]} | Predicted: {a2[i][0]:.4f} | Round: {round(a2[i][0])} | True: {y[i][0]}")

# Loss Visualization
plt.figure(figsize=(7, 4))
plt.plot(loss_history, color='tab:purple', linewidth=2)
plt.title('XOR Neural Network Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()