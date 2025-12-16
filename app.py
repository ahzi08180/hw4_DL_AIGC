import streamlit as st
from PIL import Image
from predict import load_model, predict_image
import os
import torch
import pandas as pd

st.set_page_config(page_title="Bird Species Classifier", layout="centered")

st.title("🦜 Bird Species Classification")
st.write("Upload a bird image or use a sample image to get predicted species probabilities.")

# Load model
@st.cache_resource
def load():
    return load_model("model/bird_model.pth")

model, class_names = load()

# 選擇輸入方式
option = st.radio("Choose input method:", ("Upload Image", "Use Sample Image"))

def get_predictions(image):
    # transform image
    image = image.convert("RGB")
    transform = st.session_state.get("transform", None)
    if transform is None:
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        st.session_state["transform"] = transform

    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1).squeeze()
    return probs.cpu().numpy()

def display_results(probs):
    # 將機率與類別對應
    df = pd.DataFrame({
        "Class": class_names,
        "Probability": probs
    })
    # 排序
    df = df.sort_values("Probability", ascending=False).reset_index(drop=True)
    
    st.subheader("Top Prediction")
    st.success(f"{df.loc[0, 'Class']} ({df.loc[0, 'Probability']*100:.2f}%)")
    
    st.subheader("All Class Probabilities")
    st.dataframe(df.style.format({"Probability": "{:.2%}"}))

# --- Upload Image ---
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        with st.spinner("Predicting..."):
            probs = get_predictions(image)
            display_results(probs)

# --- Use Sample Image ---
elif option == "Use Sample Image":
    sample_dir = "sample_images"
    sample_files = os.listdir(sample_dir)
    sample_file = st.selectbox("Select a sample image:", sample_files)
    if sample_file:
        image_path = os.path.join(sample_dir, sample_file)
        image = Image.open(image_path)
        st.image(image, caption=f"Sample: {sample_file}", use_column_width=True)
        with st.spinner("Predicting..."):
            probs = get_predictions(image)
            display_results(probs)
