import streamlit as st
from PIL import Image
from predict import load_model, predict_image
import os

st.set_page_config(page_title="Bird Species Classifier", layout="centered")

st.title("🦜 Bird Species Classification")
st.write("Upload a bird image or use a sample image to get the predicted species.")

# Load model
@st.cache_resource
def load():
    return load_model("model/bird_model.pth")

model, class_names = load()

# --- 選擇使用方式 ---
option = st.radio("Choose input method:", ("Upload Image", "Use Sample Image"))

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Predicting..."):
            label, confidence = predict_image(model, class_names, image)

        st.success(f"🧠 Prediction: **{label}**")
        st.info(f"Confidence: {confidence:.2%}")

elif option == "Use Sample Image":
    sample_dir = "sample_images"
    sample_files = os.listdir(sample_dir)
    sample_file = st.selectbox("Select a sample image:", sample_files)
    
    if sample_file:
        image_path = os.path.join(sample_dir, sample_file)
        image = Image.open(image_path).convert("RGB")
        st.image(image, caption=f"Sample: {sample_file}", use_column_width=True)

        with st.spinner("Predicting..."):
            label, confidence = predict_image(model, class_names, image)

        st.success(f"🧠 Prediction: **{label}**")
        st.info(f"Confidence: {confidence:.2%}")
