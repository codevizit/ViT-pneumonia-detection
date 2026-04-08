import streamlit as st
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image

# ---------------------------
# 🎨 Page Config
# ---------------------------
st.set_page_config(
    page_title="X-ray Pneumonia Detection",
    page_icon="🩺",
    layout="centered"
)

# ---------------------------
# 🎨 Custom CSS (Blue Theme)
# ---------------------------
st.markdown("""
<style>
body {
    background-color: #f4f8fb;
}
.main {
    background-color: #f4f8fb;
}
h1 {
    color: #1f4e79;
    text-align: center;
}
.stButton>button {
    background-color: #1f77b4;
    color: white;
    border-radius: 10px;
    padding: 0.5em 1em;
}
.stButton>button:hover {
    background-color: #145a86;
    color: white;
}
.result-box {
    padding: 15px;
    border-radius: 10px;
    background-color: #e6f2ff;
    text-align: center;
    font-size: 20px;
    font-weight: bold;
    color: #0b3c5d;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# 🧠 Load Model
# ---------------------------

import os


@st.cache_resource
def load_model():
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    model.classifier = torch.nn.Linear(model.config.hidden_size, 2)
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "vit_pneumonia_model_weights.pth")
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    return model, processor

model, processor = load_model()

# ---------------------------
# 🖼️ UI
# ---------------------------
st.title("🩺 X-ray Pneumonia Detection")
st.markdown("Upload a chest X-ray image to detect **Pneumonia or Normal**.")

uploaded_file = st.file_uploader("📤 Upload X-ray Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Analyze"):
        with st.spinner("Analyzing..."):
            inputs = processor(images=image, return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)
                pred = torch.argmax(outputs.logits, dim=1).item()

            label = "PNEUMONIA " if pred == 1 else "NORMAL "

        st.markdown(f'<div class="result-box">Prediction: {label}</div>', unsafe_allow_html=True)

# ---------------------------
# 📌 Footer
# ---------------------------
st.markdown("---")
st.markdown(
    "<center>Developed for AI-based Medical Diagnosis | ViT Model</center>",
    unsafe_allow_html=True
)
