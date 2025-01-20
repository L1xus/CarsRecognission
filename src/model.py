import torch
from torchvision import models
from torch import nn, optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_resnet(num_classes):
    model = models.resnet34(pretrained=True)

    # Freeze all layers except the final fully connected layer
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=0.01, momentum=0.9)
    lrscheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, threshold=1e-4)

    return model, criterion, optimizer, lrscheduler

def save_checkpoint(model, optimizer, epoch, class_to_idx, filepath):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'class_to_idx': class_to_idx,
    }, filepath)

    print(f"Checkpoint saved at epoch {epoch}")

def load_checkpoint(filepath, num_classes):
    checkpoint = torch.load(filepath)
    model, _, optimizer, _ = create_resnet(num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    class_to_idx = checkpoint['class_to_idx']
    print(f"Checkpoint loaded: Epoch {epoch}")

    return model, optimizer, epoch, class_to_idx
