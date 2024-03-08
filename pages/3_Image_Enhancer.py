import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.models import load_model

st.set_page_config(
    page_title="Image Enhancer",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)
st.title("⚡ Image Enhancer")
st.caption("Enhance the quality of images using our pre-trained model.")
st.markdown("---")

# load the model architecture without compiling
try:
    model = tf.keras.models.load_model('models/image_enhancer.h5', compile=False)
except Exception as e:
    st.error(f"Error loading the model: {str(e)}")

# manually compile the model with the desired settings
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam')
# model.compile(optimizer=optimizer, loss='your_loss_function', metrics=['your_metrics'])


def load_image(image_file):
    """Load the uploaded image file."""
    img = Image.open(image_file)
    return img

def enhance_image(model, image):
    """Enhance the image using the pre-trained model, with adjusted preprocessing."""
    
    # adjust the image to match the model's expected input size and preprocessing
    img = image.resize((64, 64))
    img_array = np.array(img) / 255.0
    img_array = img_array[np.newaxis, ...]

    enhanced_img_array = model.predict(img_array)
    enhanced_img = Image.fromarray((enhanced_img_array.squeeze() * 255).astype(np.uint8))
    return enhanced_img

st.title('Image Enhancement Application')

uploaded_file = st.file_uploader("Choose an image to upload", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = load_image(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button('Enhance Image'):
        enhanced_image = enhance_image(model, image)
        st.image(enhanced_image, caption='Enhanced Image', use_column_width=True)
        
        enhanced_image.save("enhanced_image.png")
        with open("enhanced_image.png", "rb") as file:
            btn = st.download_button(
                label="Download Enhanced Image",
                data=file,
                file_name="enhanced_image.png",
                mime="image/png"
            )
