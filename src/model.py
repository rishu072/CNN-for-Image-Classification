"""
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3,16,3)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(16,32,3)

        self.fc1 = nn.Linear(32*6*6,64)
        self.fc2 = nn.Linear(64,10)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1,32*6*6)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    """
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        # 🔹 First layer
        self.conv1 = nn.Conv2d(3, 16, 3)   # input=3 (RGB), output=16
        self.pool = nn.MaxPool2d(2,2)

        # 🔹 Second layer
        self.conv2 = nn.Conv2d(16, 32, 3)

        # 🔹 Fully connected layers
        self.fc1 = nn.Linear(32 * 6 * 6, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):

        # step 1
        x = self.pool(F.relu(self.conv1(x)))

        # step 2
        x = self.pool(F.relu(self.conv2(x)))

        # flatten
        x = x.view(-1, 32 * 6 * 6)

        # step 3
        x = F.relu(self.fc1(x))

        # output
        x = self.fc2(x)

        return x