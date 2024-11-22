import os
import streamlit as st
from PIL import Image
import torch
from model.data.transforms import rgb_to_gray, increase_color, rgb_to_lab, lab_to_rgb
from model.model import CNN, Generator
import numpy as np
import cv2

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
model_cnn.load_state_dict(torch.load("./models/cnn/best_model_after42.pt", weights_only=True, map_location=torch.device('cpu')))
model_cnn.eval()

model_gan = Generator()
model_gan.load_state_dict(torch.load("./models/gan/best.pth", weights_only=True, map_location=torch.device('cpu'))["model_generator"])
model_gan.eval()

def preprocess_image(image, target_size=(1024, 1024)):
    return image.resize(target_size, Image.LANCZOS)

# Colorization function
def colorize_image_cnn(image):
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
    L = torch.Tensor(rgb_to_lab(image).astype("float32")).to('cpu')
    L = L.permute(2, 0, 1)
    L = L[[0],...] / 50. - 1.
    L = L.unsqueeze(0)

    with torch.no_grad():
        colored = model_gan(L)
        
    colored = lab_to_rgb(L, colored.detach())[0]
    return colored


# Function to display upload and colorize page
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

        # Add a slider for brightness adjustment
        brightness_factor = st.slider("Adjust Brightness 1", 0.5, 2.0, 1.0, 0.1)

        # Layout with two columns for input and output images
        col1, col2, col3 = st.columns(3)
        url = 'https://media.istockphoto.com/id/1409329028/vector/no-picture-available-placeholder-thumbnail-icon-illustration-design.jpg?s=612x612&w=0&k=20&c=_zOuJu755g2eEUioiOUdz_mHKJQJn-tDgIAhQzyeKUQ='

        with col1:
            st.image(np_image, caption="Brightness Adjusted Image", use_container_width=True)
        
        with col2:
            # Placeholder for colorized image
            placeholder = st.empty()
            placeholder.image(url, caption="Colorized with CNN", use_container_width=True)
            
            # "Colorize" button
            if st.button("Colorize with CNN"):
                # Run the colorization model on the brightness-adjusted image
                colorized_image = colorize_image_cnn(np_image)
                colorized_image = increase_color(colorized_image, brightness_factor)
                
                # Display the colorized image
                placeholder.image(colorized_image, caption="Colorized Image", use_container_width=True)

            with col3:
                placeholder = st.empty()
                placeholder.image(url, caption="Colorized with GAN", use_container_width=True)
                
                # "Colorize" button
                if st.button("Colorize with GAN"):
                    # Run the colorization model on the brightness-adjusted image
                    colorized_image = colorize_image_gan(np_image)
                    # Display the colorized image
                    placeholder.image(colorized_image, caption="Colorized Image", use_container_width=True)


# Run the main display function
display_upload_and_colorize_page()
