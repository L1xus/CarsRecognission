import torch
from torchvision import transforms, models
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import s3fs
import os
from dotenv import load_dotenv

load_dotenv()
aws_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")

# Choose the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(model, optimizer, epoch, filepath="resnet34_car_classifier_checkpoint.pth"):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }, filepath)
    print(f"Checkpoint saved at epoch {epoch}")


def load_checkpoint(filepath, num_classes):
    checkpoint = torch.load(filepath)
    model, _, optimizer, _ = create_resnet(num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded: Epoch {epoch}")
    return model, optimizer, epoch

def create_resnet(num_classes):
    model = models.resnet34(pretrained=True)

    # Freeze the layers except for the final fully connected layer
    for param in model.parameters():
        param.requires_grad = False

    # Modify the final fully connected layer to match the number of classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Move model to the chosen device (CPU or GPU)
    model.to(device)

    # Set loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=0.01, momentum=0.9)
    lrscheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, threshold=1e-4
    )

    return model, criterion, optimizer, lrscheduler


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


def test_model(model, test_loader, device):
    model.eval()  
    test_loss = 0
    accuracy = 0

    criterion = (nn.CrossEntropyLoss()) 

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            accuracy += (predicted == labels).float().mean().item()

    return test_loss / len(test_loader), accuracy / len(test_loader)


class S3ImageDataset(Dataset):
    def __init__(self, s3_root, transform=None, aws_key=None, aws_secret=None):
        self.s3 = s3fs.S3FileSystem(anon=False, key=aws_key, secret=aws_secret)
        self.s3_root = s3_root.rstrip("/")
        self.transform = transform

        self.image_paths = []
        self.class_to_idx = {}

        print(f"Fetching class directories from: {self.s3_root}")
        class_dirs = self.s3.ls(self.s3_root)
        for cls_dir in class_dirs:
            if self.s3.isdir(cls_dir):
                class_name = os.path.basename(cls_dir).replace(" ", "_")
                self.class_to_idx[class_name] = len(self.class_to_idx)
                cls_image_paths = self.s3.glob(f"{cls_dir}/**/*.jpg")
                self.image_paths.extend(cls_image_paths)

        if not self.image_paths:
            raise ValueError(f"No images found at {self.s3_root}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        with self.s3.open(image_path, "rb") as file:
            image = Image.open(file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = os.path.basename(os.path.dirname(image_path)).replace(" ", "_")
        label = self.class_to_idx[label]

        return image, label


def classification():
    train_transforms = transforms.Compose([
        transforms.Resize((244, 244)),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((244, 244)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    s3_root = "s3://carset/cars/"

    try:
        train_data = S3ImageDataset(s3_root + "train", transform=train_transforms, aws_key=aws_key, aws_secret=aws_secret)
        test_data = S3ImageDataset(s3_root + "test", transform=test_transforms, aws_key=aws_key, aws_secret=aws_secret)

        validation_split = 0.2
        valid_size = int(validation_split * len(train_data))
        train_size = len(train_data) - valid_size
        train_data, valid_data = torch.utils.data.random_split(train_data, [train_size, valid_size])

    except Exception as e:
        print(f"Error loading data from S3: {e}")
        raise

    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    model, criterion, optimizer, lrscheduler = create_resnet(num_classes=196)

    epochs = 10
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Train the model
        train_loss, train_accuracy = train_model(model, train_loader, criterion, optimizer, device)
        print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")

        # Validate the model
        valid_loss, valid_accuracy = validate_model(model, valid_loader, criterion, device)
        print(f"Validation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_accuracy:.4f}")

        # Step the learning rate scheduler
        lrscheduler.step(valid_accuracy)

        # Save the model checkpoint
        save_checkpoint(model, optimizer, epoch)

    # Test the model
    test_loss, test_accuracy = test_model(model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    checkpoint_path = "resnet34_car_classifier_checkpoint.pth"
    num_classes = 196

    if os.path.exists(checkpoint_path):
        model, optimizer, epoch = load_checkpoint(checkpoint_path, num_classes)
        print(f"Resuming training from epoch {epoch + 1}")
    else:
        classification()
