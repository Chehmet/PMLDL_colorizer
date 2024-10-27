import os
import streamlit as st
from PIL import Image

UPLOAD_FOLDER = "uploads/"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

with open("templates/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def display_upload_and_colorize_page():
    st.markdown(open("templates/upload.html").read(), unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success("Image uploaded successfully!")
        image = Image.open(file_path)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            placeholder = st.empty()
            placeholder.image("https://via.placeholder.com/200", caption="Colorized Image", use_column_width=True)

            if st.button("Colorize"):
                colorized_image = image
                placeholder.image(colorized_image, caption="Colorized Image", use_column_width=True)

display_upload_and_colorize_page()