import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Face Mask Detection",
    layout="centered"
)

st.title("😷 Face Mask Detection System")
st.write(
    "This application classifies images into: "
    "**With Mask**, **Without Mask**, or **Incorrect Mask**."
)

# =========================
# Load Model
# =========================
@st.cache_resource
def load_model():
    checkpoint = torch.load("best_model.pt", map_location="cpu")

    classes = checkpoint["classes"]
    image_size = checkpoint["image_size"]

    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(classes))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ])

    return model, classes, transform

model, CLASSES, transform = load_model()

# =========================
# Upload Image
# =========================
uploaded_file = st.file_uploader(
    "Upload an image", type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    x = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).numpy()[0]
        pred_idx = np.argmax(probs)

    st.subheader("Prediction")
    st.write(f"**Class:** {CLASSES[pred_idx]}")
    st.write(f"**Confidence:** {probs[pred_idx]:.2f}")

    st.bar_chart(
        {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}
    )
