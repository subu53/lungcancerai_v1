import streamlit as st
from PIL import Image
import io
import base64

from app.model import load_model, predict_image
from app.explain import generate_gradcam_base64

st.set_page_config(page_title="Lung AI V1", layout="wide")
st.title("Lung AI V1")
st.caption("CT image classification prototype: Normal / Benign / Malignant")

@st.cache_resource
def get_model():
    return load_model()

model, class_names, device = get_model()

uploaded = st.file_uploader("Upload a CT image", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Analyze"):
        try:
            pred_class, pred_idx, confidence, image_tensor = predict_image(
                image, model, class_names, device
            )

            heatmap_b64 = generate_gradcam_base64(model, image_tensor, pred_idx)

            st.subheader("Prediction")
            st.write("Class:", pred_class)
            st.write("Confidence:", round(confidence, 4))
            st.warning("Research prototype only. Not for clinical diagnosis.")

            if heatmap_b64:
                heatmap = Image.open(io.BytesIO(base64.b64decode(heatmap_b64)))
                st.image(heatmap, caption="Grad-CAM heatmap", use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
