import torch
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")

    return transform(image).unsqueeze(0)

def predict_image(model, image_tensor, class_to_idx):
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor.to(device))
        probabilities = torch.softmax(outputs, dim=1)
        predicted_idx = probabilities.argmax(dim=1).item()
        predicted_label = list(class_to_idx.keys())[list(class_to_idx.values()).index(predicted_idx)]
        confidence = probabilities[0, predicted_idx].item()

    return predicted_label, confidence
