import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.transforms import v2
import matplotlib.pyplot as plt

# 1. SETUP CONFIGURATION & DATA LOADING

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
epochs = 5  # Kept low for demonstration execution speeds

# Setup transforms using the newer v2 API
transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# For MNIST, switch to torchvision.datasets.MNIST and change input channels to 1
print("Downloading and loading CIFAR-10 dataset")
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

# 2. DEFINE THE FLEXIBLE DEEP CNN ARCHITECTURE

class DeepCNN(nn.Module):
    def __init__(self, num_layers, in_channels=3, num_classes=10):
        super(DeepCNN, self).__init__()

        layers = []
        current_channels = in_channels
        base_channels = 72

        # Build consecutive convolutional blocks based on the dynamic layer count
        for i in range(num_layers):
            # Maintain spatial size with padding='same' to allow arbitrary configurations
            layers.append(nn.Conv2d(current_channels, base_channels, kernel_size=3, padding='same'))
            layers.append(nn.ReLU())
            current_channels = base_channels

        # Group conv layers together
        self.conv_blocks = nn.Sequential(*layers)

        # Downsample features before passing to linear classifiers
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.pool(x)
        x = self.fc(x)
        return x

# 3. TRAINING ENGINE LOOP

def train_model(num_layers):
    print(f"\nTraining CNN with {num_layers} Convolutional Layers")
    model = DeepCNN(num_layers=num_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    # Using plain SGD without momentum to cleanly highlight the degradation effect
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    final_train_loss = 0.0

    for epoch in range(epochs):
        running_loss = 0.0
        model.train()

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(trainloader)
        print(f"Epoch {epoch + 1}/{epochs} - Training Loss: {epoch_loss:.4f}")
        final_train_loss = epoch_loss

    return final_train_loss

# 4. EXPERIMENTATION & PLOTTING ERROR VS LAYERS
# Test depths from a shallow 2-layer network up to a very deep 30-layer system
layer_options = [2, 5, 10, 20, 30]
train_errors = []

for layers in layer_options:
    final_loss = train_model(layers)
    train_errors.append(final_loss)

# Generate the convergence analysis plot
plt.figure(figsize=(8, 5))
plt.plot(layer_options, train_errors, marker='o', color='red', linestyle='-', linewidth=2)
plt.title('CNN Degradation Analysis: Training Error vs. Number of Layers')
plt.xlabel('Number of Convolutional Layers')
plt.ylabel('Final Training Cross-Entropy Loss')
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('train_error_vs_layers.png')
plt.show()