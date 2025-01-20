import os
from src.dataset import S3ImageDataset
from src.model import create_resnet, load_checkpoint
from src.train import train_model, validate_model
from src.test import test_model
from src.predict import preprocess_image, predict_image
from src.transforms import get_train_transforms, get_test_transforms
from src.config import *

import torch
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # Load dataset
    train_transforms = get_train_transforms()
    test_transforms = get_test_transforms()

    train_data = S3ImageDataset(S3_ROOT + "train", transform=train_transforms, aws_key=AWS_ACCESS_KEY_ID, aws_secret=AWS_SECRET_ACCESS_KEY)
    test_data = S3ImageDataset(S3_ROOT + "test", transform=test_transforms, aws_key=AWS_ACCESS_KEY_ID, aws_secret=AWS_SECRET_ACCESS_KEY)

    # Split train/validation data
    validation_split = 0.2
    valid_size = int(validation_split * len(train_data))
    train_size = len(train_data) - valid_size
    train_data, valid_data = torch.utils.data.random_split(train_data, [train_size, valid_size])

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # Load or create model
    if os.path.exists(CHECKPOINT_PATH):
        model, _, _ = load_checkpoint(CHECKPOINT_PATH, NUM_CLASSES)
    else:
        model, criterion, optimizer, lrscheduler = create_resnet(NUM_CLASSES)

        for epoch in range(EPOCHS):
            train_loss, train_accuracy = train_model(model, train_loader, criterion, optimizer, device)
            valid_loss, valid_accuracy = validate_model(model, valid_loader, criterion, device)
            print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Valid Accuracy: {valid_accuracy}")
            lrscheduler.step(valid_accuracy)

    # Test
    test_loss, test_accuracy = test_model(model, test_loader, device)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

    # Predict on a single image
    image_path = "e39.jpg"
    class_to_idx = train_data.dataset.class_to_idx
    transform = get_test_transforms()
    image_tensor = preprocess_image(image_path, transform)
    predicted_label, confidence = predict_image(model, image_tensor, class_to_idx)
    print(f"Predicted: {predicted_label}, Confidence: {confidence}")
