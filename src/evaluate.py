import torch

def evaluate(model,testloader):

    correct = 0
    total = 0

    with torch.no_grad():
        for images,labels in testloader:
            outputs = model(images)
            _,predicted = torch.max(outputs,1)

            total += labels.size(0)
            correct += (predicted==labels).sum().item()

    print("Accuracy:",100*correct/total)