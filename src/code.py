from torchvision import datasets, transforms
import s3fs
import os
from dotenv import load_dotenv

load_dotenv()
aws_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")


def classification():
    s3_root = f"s3://carset/cars/"
    s3 = s3fs.S3FileSystem(anon=False, key=aws_key, secret=aws_secret)


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

    validation_transforms = transforms.Compose([
        transforms.Resize((244, 244)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    try:
        train_data = datasets.ImageFolder(s3_root + "train", transform=train_transforms)
        test_data = datasets.ImageFolder(s3_root + "test", transform=test_transforms)
        valid_data = datasets.ImageFolder(s3_root + "valid", transform=validation_transforms)

        print(f"Number of training samples: {len(train_data)}")
        print(f"Number of test samples: {len(test_data)}")
        print(f"Number of validation samples: {len(valid_data)}")

    except Exception as e:
        print(f"Error loading data from S3: {e}")
        raise

    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=32, shuffle=True)
