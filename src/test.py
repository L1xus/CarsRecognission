import torch
from torch import nn

def test_model(model, test_loader, device):
    model.eval()
    test_loss = 0
    accuracy = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            accuracy += (predicted == labels).float().mean().item()

    return test_loss / len(test_loader), accuracy / len(test_loader)
