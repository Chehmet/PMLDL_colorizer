import os
import streamlit as st
from PIL import Image
import torch
from model.data.transforms import rgb_to_gray, rgb_to_lab, lab_to_rgb
from model.model import CNN, Generator
import numpy as np
import cv2
import gdown

import warnings
warnings.filterwarnings('ignore')

# Load custom CSS
with open("templates/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Define upload folder
UPLOAD_FOLDER = "uploads/"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Cached model loading functions
@st.cache_resource
def load_cnn_model():
    model_cnn = CNN()
    model_cnn.load_state_dict(torch.load("./models/cnn/best_model_after42.pt", map_location=torch.device('cpu')))
    model_cnn.eval()
    return model_cnn

@st.cache_resource
def load_gan_model():
    url = "https://drive.google.com/uc?id=19awWsef7oDQxMFGN7_qN2Cd0pQE1E6Jl"
    output = "./models/gan/best.pth"
    # Ensure the models directory exists
    if not os.path.exists("./models/gan/"):
        os.makedirs("./models/gan/")
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
    model_gan = Generator()
    checkpoint = torch.load(output, map_location=torch.device('cpu'))
    model_gan.load_state_dict(checkpoint["model_generator"])
    model_gan.eval()
    return model_gan

# Load models
model_cnn = load_cnn_model()
model_gan = load_gan_model()

def preprocess_image(image, target_size=(1024, 1024)):
    return image.resize(target_size, Image.LANCZOS)

def adjust_brightness(image: np.ndarray, brightness_factor: float) -> np.ndarray:
    """Adjust the brightness of the image."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 2] = hsv[..., 2] * brightness_factor
    hsv[..., 2] = np.clip(hsv[..., 2], 0, 255)
    bright_image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return bright_image

def colorize_image_cnn(image, brightness_factor=1.0):
    if image.ndim == 2 or image.shape[-1] != 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB

    # Transform and prepare the input image
    gray = torch.Tensor(rgb_to_gray(image)).to('cpu')
    gray = gray / 255
    gray = gray.permute(2, 0, 1)  # Reorder to (C, H, W) format
    gray = gray.unsqueeze(0)  # Add batch dimension

    # Perform colorization
    with torch.no_grad():
        colored = model_cnn(gray)[0]  # Remove batch dimension after prediction

    # Convert tensor to a format suitable for displaying
    colored = colored.permute(1, 2, 0).numpy()  # Convert to (H, W, C) format
    colored = (colored * 255).astype(np.uint8)  # Scale back to image format

    # Adjust brightness
    colored = adjust_brightness(colored, brightness_factor)

    return colored

def colorize_image_gan(image, brightness_factor=1.0):
    if image.ndim == 2 or image.shape[-1] != 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB

    L = torch.Tensor(rgb_to_lab(image).astype("float32")).to('cpu')
    L = L.permute(2, 0, 1)
    L = L[[0], ...] / 50. - 1.
    L = L.unsqueeze(0)

    with torch.no_grad():
        colored = model_gan(L)

    colored = lab_to_rgb(L, colored.detach())[0]
    colored = (colored * 255).astype(np.uint8)  # Scale back to image format

    # Adjust brightness
    colored = adjust_brightness(colored, brightness_factor)

    return colored

def display_upload_and_colorize_page():
    st.markdown(open("templates/upload.html").read(), unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        # Save and display uploaded image
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success("Image uploaded successfully!")
        image = Image.open(file_path)
        image = preprocess_image(image)
        np_image = np.array(image)  # Convert to NumPy array

        # Initialize session state for storing results
        if "colorized_cnn" not in st.session_state:
            st.session_state["colorized_cnn"] = None
        if "colorized_gan" not in st.session_state:
            st.session_state["colorized_gan"] = None

        # Add a slider for brightness adjustment
        brightness_factor = st.slider("Adjust Brightness", 0.5, 2.0, 1.0, 0.1)

        # Layout with three columns for input and colorized images
        col1, col2, col3 = st.columns(3)
        placeholder_image_url = 'https://media.istockphoto.com/id/1409329028/vector/no-picture-available-placeholder-thumbnail-icon-illustration-design.jpg?s=612x612&w=0&k=20&c=_zOuJu755g2eEUioiOUdz_mHKJQJn-tDgIAhQzyeKUQ='

        with col1:
            st.image(np_image, caption="Uploaded Image")

        with col2:
            if st.session_state["colorized_cnn"] is not None:
                st.image(st.session_state["colorized_cnn"], caption="Colorized with CNN")
            else:
                st.image(placeholder_image_url, caption="Colorized with CNN")

            if st.button("Colorize with CNN"):
                colorized_cnn = colorize_image_cnn(np_image, brightness_factor)
                st.session_state["colorized_cnn"] = colorized_cnn

        with col3:
            if st.session_state["colorized_gan"] is not None:
                st.image(st.session_state["colorized_gan"], caption="Colorized with GAN")
            else:
                st.image(placeholder_image_url, caption="Colorized with GAN")

            if st.button("Colorize with GAN"):
                colorized_gan = colorize_image_gan(np_image, brightness_factor)
                st.session_state["colorized_gan"] = colorized_gan

# Run the main display function
if __name__ == '__main__':
    display_upload_and_colorize_page()
