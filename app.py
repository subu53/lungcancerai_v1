from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torch

from model import load_model, predict_image
from explain import generate_gradcam_base64

app = FastAPI(title="Lung AI V1")

model, class_names, device = load_model()

@app.get("/")
def root():
    return {"message": "Lung AI V1 API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    pred_class, confidence, image_tensor = predict_image(image, model, class_names, device)
    heatmap_b64 = generate_gradcam_base64(model, image_tensor, pred_class)

    return JSONResponse({
        "predicted_class": pred_class,
        "confidence": float(confidence),
        "heatmap_base64": heatmap_b64,
        "warning": "Research prototype only. Not for clinical diagnosis."
    })