import torch

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0
    accuracy = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        accuracy += (predicted == labels).float().mean().item()

    return train_loss / len(train_loader), accuracy / len(train_loader)

def validate_model(model, valid_loader, criterion, device):
    model.eval()
    valid_loss = 0
    accuracy = 0

    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            accuracy += (predicted == labels).float().mean().item()

    return valid_loss / len(valid_loader), accuracy / len(valid_loader)
