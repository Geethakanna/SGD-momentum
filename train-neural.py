import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

inputSize = 784  # 28x28 images
hiddenSize = 128
numClasses = 10
numEpochs = 5
batchSize = 64
learningRate = 0.01
momentumValue = 0.9

transform = transforms.ToTensor()
trainDataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
testDataset = datasets.MNIST(root='./data', train=False, transform=transform)

trainLoader = DataLoader(dataset=trainDataset, batch_size=batchSize, shuffle=True)
testLoader = DataLoader(dataset=testDataset, batch_size=batchSize, shuffle=False)

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(inputSize, hiddenSize)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hiddenSize, numClasses)

    def forward(self, x):
        x = x.view(-1, inputSize)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def train(model, optimizer, dataLoader):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for epoch in range(numEpochs):
        totalLoss = 0
        for images, labels in dataLoader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            totalLoss += loss.item()
        print(f"Epoch [{epoch + 1}/{numEpochs}], Loss: {totalLoss / len(dataLoader):.4f}")

def test(model, dataLoader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataLoader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')
    return accuracy

print("\nTraining with SGD (no momentum)")
modelSgd = NeuralNet().to(device)
optimizerSgd = optim.SGD(modelSgd.parameters(), lr=learningRate)
train(modelSgd, optimizerSgd, trainLoader)
accuracySgd = test(modelSgd, testLoader)

print("\nTraining with SGD with Momentum")
modelMomentum = NeuralNet().to(device)
optimizerMomentum = optim.SGD(modelMomentum.parameters(), lr=learningRate, momentum=momentumValue)
train(modelMomentum, optimizerMomentum, trainLoader)
accuracyMomentum = test(modelMomentum, testLoader)

print("\nComparison:")
print(f"SGD Accuracy: {accuracySgd:.2f}%")
print(f"SGD with Momentum Accuracy: {accuracyMomentum:.2f}%")
