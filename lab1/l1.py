import numpy as np
import matplotlib.pyplot as plt

# 100 equally spaced values between -10 and 10
z = np.linspace(-10, 10, 100)

#Sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

#Tanh
def tanh(z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

def tanh_derivative(z):
    return 1 - tanh(z)**2

#ReLU
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return np.where(z > 0, 1.0, 0.0)


#Leaky ReLU
def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

def leaky_relu_derivative(z, alpha=0.01):
    return np.where(z > 0, 1.0, alpha)


#Softmax
def softmax(z):

    e_z = np.exp(z - np.max(z))
    return e_z / np.sum(e_z)
# Sigmoid Plot

plt.plot(z, sigmoid(z))
plt.title("Sigmoid Function")
plt.xlabel("z")
plt.ylabel("Output")
plt.show()
# Tanh Plot

plt.plot(z, tanh(z))
plt.title("Tanh Function")
plt.xlabel("z")
plt.ylabel("Output")
plt.show()

#ReLU Plot

plt.plot(z, relu(z))
plt.title("ReLU Function")
plt.xlabel("z")
plt.ylabel("Output")
plt.show()
# Leaky ReLU Plot

plt.plot(z, leaky_relu(z))
plt.title("Leaky ReLU Function")
plt.xlabel("z")
plt.ylabel("Output")
plt.show()