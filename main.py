import os
import streamlit as st
from PIL import Image
import torch
from model.data.transforms import rgb_to_gray, increase_color, rgb_to_lab, lab_to_rgb
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

# Load the PictureColorizer model
model_cnn = CNN()
model_cnn.load_state_dict(torch.load("./models/cnn/best_model_after42.pt", map_location=torch.device('cpu')))
# print(state_dict.keys())
model_cnn.eval()

url = "https://drive.google.com/uc?id=19awWsef7oDQxMFGN7_qN2Cd0pQE1E6Jl"
output = "./models/gan/best.pth"
gdown.download(url, output, quiet=False)

# Step 2: Load the model
model_gan = Generator()
checkpoint = torch.load(output, map_location=torch.device('cpu'))
model_gan.load_state_dict(checkpoint["model_generator"])
model_gan.eval()
def preprocess_image(image, target_size=(1024, 1024)):
    return image.resize(target_size, Image.LANCZOS)

# Colorization function
def colorize_image_cnn(image):
    if image.ndim == 2 or image.shape[-1] != 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB

    # Transform and prepare the input image
    gray = torch.Tensor(rgb_to_gray(image)).to('cpu')
    gray = gray/255
    gray = gray.permute(2, 0, 1)  # Reorder to (C, H, W) format
    gray = gray.unsqueeze(0)  # Add batch dimension

    # Perform colorization
    with torch.no_grad():
        colored = model_cnn(gray)[0]  # Remove batch dimension after prediction

    # Convert tensor to a format suitable for displaying
    colored = colored.permute(1, 2, 0).numpy()  # Convert to (H, W, C) format
    colored = (colored * 255).astype(np.uint8)  # Scale back to image format
    return colored


def colorize_image_gan(image):
    if image.ndim == 2 or image.shape[-1] != 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB

    L = torch.Tensor(rgb_to_lab(image).astype("float32")).to('cpu')
    L = L.permute(2, 0, 1)
    L = L[[0], ...] / 50. - 1.
    L = L.unsqueeze(0)

    with torch.no_grad():
        colored = model_gan(L)

    colored = lab_to_rgb(L, colored.detach())[0]
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
        brightness_factor = st.slider("Adjust Brightness 1", 0.5, 2.0, 1.0, 0.1)

        # Layout with three columns for input and colorized images
        col1, col2, col3 = st.columns(3)
        url = 'https://media.istockphoto.com/id/1409329028/vector/no-picture-available-placeholder-thumbnail-icon-illustration-design.jpg?s=612x612&w=0&k=20&c=_zOuJu755g2eEUioiOUdz_mHKJQJn-tDgIAhQzyeKUQ='

        with col1:
            st.image(np_image, caption="Uploaded Image")

        with col2:
            if st.session_state["colorized_cnn"] is not None:
                st.image(st.session_state["colorized_cnn"], caption="Colorized with CNN")
            else:
                st.image(url, caption="Colorized with CNN")

            if st.button("Colorize with CNN"):
                colorized_cnn = colorize_image_cnn(np_image)
                colorized_cnn = increase_color(colorized_cnn, brightness_factor)
                st.session_state["colorized_cnn"] = colorized_cnn

        with col3:
            if st.session_state["colorized_gan"] is not None:
                st.image(st.session_state["colorized_gan"], caption="Colorized with GAN")
            else:
                st.image(url, caption="Colorized with GAN")

            if st.button("Colorize with GAN"):
                colorized_gan = colorize_image_gan(np_image)
                st.session_state["colorized_gan"] = colorized_gan

# Run the main display function
display_upload_and_colorize_page()
