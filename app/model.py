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
