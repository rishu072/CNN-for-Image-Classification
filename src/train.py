"""
import torch
import torch.nn as nn
import torch.optim as optim
from src.model import CNN

def train_model(trainloader):

    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    losses = []

    for epoch in range(3):

        running_loss = 0

        for images,labels in trainloader:

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs,labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        losses.append(running_loss)
        print("Epoch:",epoch,"Loss:",running_loss)

    return model,losses
    """

import torch
import torch.nn as nn
import torch.optim as optim
from src.model import CNN

def train_model(trainloader):

    model = CNN()

    # loss function
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    losses = []

    # training loop
    for epoch in range(3):   # small training

        running_loss = 0

        for images, labels in trainloader:

            # reset gradients
            optimizer.zero_grad()

            # forward pass
            outputs = model(images)

            # loss calculate
            loss = criterion(outputs, labels)

            # backward (learning)
            loss.backward()

            # update weights
            optimizer.step()

            running_loss += loss.item()

        losses.append(running_loss)

        print("Epoch:", epoch, "Loss:", running_loss)

    return model, losses