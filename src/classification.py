import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import s3fs
import os
from dotenv import load_dotenv

load_dotenv()
aws_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")

class S3ImageDataset(Dataset):
    def __init__(self, s3_root, transform=None, aws_key=None, aws_secret=None):
        self.s3 = s3fs.S3FileSystem(anon=False, key=aws_key, secret=aws_secret)
        self.s3_root = s3_root.rstrip("/")
        self.transform = transform

        self.image_paths = self.s3.glob(f"{self.s3_root}/**/*.jpg")
        if not self.image_paths:
            raise ValueError(f"No images found at {self.s3_root}.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        with self.s3.open(image_path, 'rb') as file:
            image = Image.open(file).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = os.path.basename(os.path.dirname(image_path))
        return image, label


def classification():
    s3_root = "s3://carset/cars/"

    train_transforms = transforms.Compose([
        transforms.Resize((244, 244)),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((244, 244)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    try:
        train_data = S3ImageDataset(s3_root + "train", transform=train_transforms, aws_key=aws_key, aws_secret=aws_secret)
        test_data = S3ImageDataset(s3_root + "test", transform=test_transforms, aws_key=aws_key, aws_secret=aws_secret)

        validation_split = 0.2
        valid_size = int(validation_split * len(train_data))
        train_size = len(train_data) - valid_size

        train_data, valid_data = torch.utils.data.random_split(train_data, [train_size, valid_size])

        print(f"Number of training samples: {len(train_data)}")
        print(f"Number of test samples: {len(test_data)}")
        print(f"Number of validation samples: {len(valid_data)}")

    except Exception as e:
        print(f"Error loading data from S3: {e}")
        raise

    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=32, shuffle=True)

    for images, labels in train_loader:
        print(f"Batch size: {images.size(0)}")
        print(f"Image shape: {images.shape}")
        print(f"Labels: {labels}")
        break
