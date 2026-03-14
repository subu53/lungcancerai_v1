import streamlit as st
import requests
from PIL import Image
import base64
import io

st.set_page_config(page_title="Lung AI V1", layout="wide")
st.title("Lung AI V1")
st.caption("CT image classification prototype: Normal / Benign / Malignant")

uploaded = st.file_uploader("Upload a CT image", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Analyze"):
        files = {"file": uploaded.getvalue()}
        response = requests.post("http://127.0.0.1:8000/predict", files=files)

        if response.status_code == 200:
            data = response.json()
            st.subheader("Prediction")
            st.write("Class:", data["predicted_class"])
            st.write("Confidence:", round(data["confidence"], 4))
            st.warning(data["warning"])

            if data["heatmap_base64"]:
                heatmap = Image.open(io.BytesIO(base64.b64decode(data["heatmap_base64"])))
                st.image(heatmap, caption="Grad-CAM heatmap", use_container_width=True)
        else:
            st.error("Prediction failed.")