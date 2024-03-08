import streamlit as st
import tensorflow as tf
from keras.models import load_model
from keras.utils import load_img, img_to_array
import numpy as np
from PIL import Image
import io

st.set_page_config(
    page_title="Image Denoiser",
    page_icon="🖼️",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)
st.title("🖼️ Image Denoiser")
st.caption("Upload a noisy image and remove noise using our pre-trained model.")
st.markdown("---")

def denoise_image(model, img_array):
    denoised_img = model.predict(img_array)
    return denoised_img[0]

loaded_model = load_model('models/denoising_model.h5')

uploaded_file = st.file_uploader("Upload a Noisy Image", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:
    # load and preprocess the uploaded image
    img = Image.open(uploaded_file)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # display the original image
    st.image(img, caption="Original Image", use_column_width=True)

    if st.button("Denoise Image"):
        denoised_img = denoise_image(loaded_model, img_array)

        # display the denoised image
        st.image(denoised_img, caption="Denoised Image", use_column_width=True)

        if st.button("Download Denoised Image"):
            denoised_pil_img = Image.fromarray((denoised_img * 255).astype(np.uint8))
            buffer = io.BytesIO()
            denoised_pil_img.save(buffer, format="JPEG")

            # download the denoised image
            st.download_button(
                label="Download Denoised Image",
                on_click=lambda: st.write(buffer.getvalue()),
                file_name="denoised_image.jpg",
                key="download_denoised_image",
                help="Click here to download the denoised image.",
                mime="image/jpeg"
            )
