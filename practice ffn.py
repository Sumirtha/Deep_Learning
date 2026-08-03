import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import v2

class neuralnetwork(nn.module):
    def __init__(self):
        super(neuralnetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(28*28, 512)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(512, 512)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(512, 10)


    def forward(self, x):
        x = self.flatten(x)
        x = self. linear1(x)
        x = self.relu1(x)
        x = self. linear2(x)
        x = self.relu2(x)
        x = self. linear3(x)
        return x

def load_data():
    training_data = datasets.FashionMNIST(root=',/data', train=True, download=True, transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]))
    test_data = datasets.FashionMNIST(root= './data', train =True, download=True, transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]))
    return training_data, test_data

def train(model, mydataloader, optimizer, loss_fn, epochs, device):
    size = len(mydataloader.dataset)
    for epoch in range(epochs):
        model.train()
        for batch, (X, y) in enumerate(mydataloader):
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(model, mydataloader, device, loss_fn):
    size = len(mydataloader.dataset)
    num_batches = len(mydataloader)
    model.eval()

    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in mydataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    print(f"Test error {correct * 100:.2f}, Avg loss: ({test_loss * 100:.2f}) ")

def main():
    training_data, test_data = load_data()
    batch_size = 64
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    for X, y in train_dataloader:
        print(f"Shape of X: {X.shape}, y shape: {y.shape}")
        break

    device  = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"using device: {device}")
    model = neuralnetwork().to(device)
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train(model, train_dataloader, optimizer, loss_fn, 10, device=device)
    test(model, test_dataloader, device, loss_fn)

    print("Finished Training")

if __name__ == "__main__":
    main()