import json
import torch
import torch.nn as nn
from torchvision import models, transforms

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 3)

    model.load_state_dict(torch.load("models/efficientnet_lung_v1.pth", map_location=device))
    model.to(device)
    model.eval()

    with open("models/class_names.json", "r") as f:
        class_names = json.load(f)

    return model, class_names, device

def predict_image(image, model, class_names, device):
    tfm = get_transforms()
    image_tensor = tfm(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()

    pred_class = class_names[pred_idx]

    return pred_class, pred_idx, confidence, image_tensor
