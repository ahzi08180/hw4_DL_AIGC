import streamlit as st
from PIL import Image
from predict import load_model, predict_image

st.set_page_config(page_title="Bird Species Classifier", layout="centered")

st.title("🦜 Bird Species Classification")
st.write("Upload a bird image and let AI identify the species.")

# Load model
@st.cache_resource
def load():
    return load_model("model/bird_model.pth")

model, class_names = load()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Predicting..."):
        label, confidence = predict_image(model, class_names, image)

    st.success(f"🧠 Prediction: **{label}**")
    st.info(f"Confidence: {confidence:.2%}")
