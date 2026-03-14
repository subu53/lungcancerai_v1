from pathlib import Path
import json
import torch
import torch.nn as nn
from torchvision import models, transforms

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

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

    model_path = MODEL_DIR / "efficientnet_lung_v1.pth"
    class_path = MODEL_DIR / "class_names.json"

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    with open(class_path, "r") as f:
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
