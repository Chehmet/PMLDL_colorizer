import os
import streamlit as st
from PIL import Image
import torch
from model.data.transforms import transform
from model.model import PictureColorizer
import numpy as np

# Load custom CSS
with open("templates/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Define upload folder
UPLOAD_FOLDER = "uploads/"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the PictureColorizer model
model = PictureColorizer()
model.load_state_dict(torch.load("./models/best.pt", weights_only=True))
model.eval()

# Colorization function
def colorize_image(image):
    # Transform and prepare the input image
    gray = torch.Tensor(transform(image))
    gray = gray.permute(2, 0, 1)  # Reorder to (C, H, W) format
    gray = gray.unsqueeze(0)  # Add batch dimension
    
    # Perform colorization
    with torch.no_grad():
        colored = model(gray)[0]  # Remove batch dimension after prediction

    # Convert tensor to a format suitable for displaying
    colored = colored.permute(1, 2, 0).numpy()  # Convert to (H, W, C) format
    colored = (colored * 255).astype(np.uint8)  # Scale back to image format
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
        
        # Layout with two columns for input and output images
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            # Placeholder for colorized image
            placeholder = st.empty()
            placeholder.image("https://via.placeholder.com/200", caption="Colorized Image", use_column_width=True)
            
            # "Colorize" button
            if st.button("Colorize"):
                # Convert the PIL image to a NumPy array
                np_image = np.array(image)
                
                # Run the colorization model
                colorized_image = colorize_image(np_image)
                
                # Display the colorized image
                placeholder.image(colorized_image, caption="Colorized Image", use_column_width=True)

# Run the main display function
display_upload_and_colorize_page()
