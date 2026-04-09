from src.train import train_model
from src.evaluate import evaluate
from src.model import CNN
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

model = CNN()
print(model)

# dataset transform
transform = transforms.ToTensor()

# dataset load
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)

# dataloader
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

dataiter = iter(trainloader)
images, labels = next(dataiter)

plt.imshow(np.transpose(images[0].numpy(), (1,2,0)))
plt.title(labels[0].item())
plt.show()

# train
model,losses = train_model(trainloader)

# test
evaluate(model,testloader)

# graph
plt.plot(losses)
plt.title("Loss")
plt.show()

from src.train import train_model

model, losses = train_model(trainloader)
correct = 0
total = 0

with torch.no_grad():
    for images, labels in testloader:

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print("Accuracy:", 100 * correct / total)