import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# =========================
# 1. Page Config & Title
# =========================
st.set_page_config(
    page_title="Mask Detection System",
    page_icon="😷",
    layout="centered"
)

st.title("Face Mask Classification System")
st.markdown("""
**Capstone Project Implementation** This system classifies images into **Mask**, **No Mask**, or **Incorrect Mask**.
*Privacy Mode is ENABLED: Identities are automatically obscured.*
""")

# =========================
# 2. Privacy & Helper Functions
# =========================
def anonymize_face_visual(img_pil: Image.Image) -> np.ndarray:
    """
    Applies pixelation to the upper half of the image to demonstrate 
    Privacy Protection / Responsible AI compliance.
    """
    img_array = np.array(img_pil)
    H, W, C = img_array.shape
    
    # Increase coverage to 50% to ensure eyes are covered
    top_h_limit = int(H * 0.50)
    block_size = 25
    
    ano_img = img_array.copy()
    
    # Simple pixelation logic
    for r in range(0, top_h_limit, block_size):
        for c in range(0, W, block_size):
            r_end = min(r + block_size, top_h_limit)
            c_end = min(c + block_size, W)
            # Safe slice handling
            block = ano_img[r:r_end, c:c_end, :]
            if block.size > 0:
                avg_color = np.mean(block, axis=(0, 1))
                ano_img[r:r_end, c:c_end, :] = avg_color
                
    return ano_img

# =========================
# 3. Load Model
# =========================
@st.cache_resource
def load_model_pipeline():
    # Load checkpoint
    try:
        checkpoint = torch.load("best_model.pt", map_location="cpu")
    except FileNotFoundError:
        st.error("Model file 'best_model.pt' not found. Please upload it.")
        return None, None, None

    # Extract correct keys based on training script
    classes = checkpoint["classes"]
    config = checkpoint["config"] 
    # Fallback to 224 if not found, but it should be there
    image_size = config.get("IMAGE_SIZE", 224) 

    # Re-build architecture
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(classes))
    
    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return model, classes, transform

model, CLASSES, transform = load_model_pipeline()

# =========================
# 4. Main Application Logic
# =========================
if model is not None:
    uploaded_file = st.file_uploader(
        "Upload an image (JPG/PNG)", type=["jpg", "png", "jpeg"]
    )

    if uploaded_file:
        # Load and preprocess
        image = Image.open(uploaded_file).convert("RGB")
        
        # --- ETHICAL DISPLAY ---
        # Display the ANONYMIZED version, not the raw face
        safe_image_array = anonymize_face_visual(image)
        st.image(safe_image_array, caption="Processed Image (Identity Protected)", use_container_width=True)
        # -----------------------

        # Inference
        x = transform(image).unsqueeze(0) # Add batch dim

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).numpy()[0]
            pred_idx = np.argmax(probs)
            confidence = probs[pred_idx]

        # Results Display
        st.divider()
        st.subheader("Analysis Results")
        
        # Dynamic color for result
        res_color = "green" if CLASSES[pred_idx] == "with_mask" else "red"
        if CLASSES[pred_idx] == "incorrect_mask": res_color = "orange"
        
        st.markdown(f"### Prediction: :{res_color}[{CLASSES[pred_idx]}]")
        st.write(f"**Confidence:** {confidence:.2%}")

        # Visualization
        st.write("### Probability Distribution")
        chart_data = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}
        st.bar_chart(chart_data)
        
        # Ethical Footer
        st.info("Note: This system automatically blurs the upper face region to protect individual privacy in compliance with Ethical AI standards.")
