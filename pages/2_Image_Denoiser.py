import streamlit as st
import tensorflow as tf
from keras.models import load_model
from keras.utils import load_img, img_to_array
import numpy as np
from PIL import Image
import io

# Function to denoise the image using the loaded model
def denoise_image(model, img_array):
    denoised_img = model.predict(img_array)
    return denoised_img[0]

# Load the pre-trained denoising model
loaded_model = tf.keras.models.load_model('models/denoising_model.h5')

# Streamlit app
st.title("Image Denoising App")

# Upload image through Streamlit
uploaded_file = st.file_uploader("Upload a Noisy Image", type=["jpg", "jpeg", "png"])

# Check if an image has been uploaded
if uploaded_file is not None:
    # Load and preprocess the uploaded image
    img = Image.open(uploaded_file)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Display the original image
    st.image(img, caption="Original Image", use_column_width=True)

    # Button to perform denoising
    if st.button("Denoise Image"):
        # Perform denoising
        denoised_img = denoise_image(loaded_model, img_array)

        # Display the denoised image
        st.image(denoised_img, caption="Denoised Image", use_column_width=True)

        # Button to download the denoised image
        if st.button("Download Denoised Image"):
            # Convert the NumPy array to a PIL Image
            denoised_pil_img = Image.fromarray((denoised_img * 255).astype(np.uint8))

            # Save the PIL Image to a BytesIO buffer
            buffer = io.BytesIO()
            denoised_pil_img.save(buffer, format="JPEG")

            # Download the denoised image
            st.download_button(
                label="Download Denoised Image",
                on_click=lambda: st.write(buffer.getvalue()),
                file_name="denoised_image.jpg",
                key="download_denoised_image",
                help="Click here to download the denoised image.",
                mime="image/jpeg"
            )
