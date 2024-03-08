import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.models import load_model

# Load the model architecture without compiling
try:
    model = tf.keras.models.load_model('models/image_enhancer.h5', compile=False)
except Exception as e:
    st.error(f"Error loading the model: {str(e)}")

# Manually compile the model with the desired settings
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam')  # Adjust the parameters accordingly
#model.compile(optimizer=optimizer, loss='your_loss_function', metrics=['your_metrics'])


def load_image(image_file):
    """Load the uploaded image file."""
    img = Image.open(image_file)
    return img

def enhance_image(model, image):
    """Enhance the image using the pre-trained model, with adjusted preprocessing."""
    
    # Adjust the image to match the model's expected input size and preprocessing
    # This is a placeholder; adjust these values based on your model's requirements
    img = image.resize((64, 64))  # Resize according to your model's expected input
    
    # Apply any specific preprocessing needed for your model here
    img_array = np.array(img) / 255.0  # Normalize if your model expects this range
    
    img_array = img_array[np.newaxis, ...]  # Add a batch dimension

    # Use the model to enhance the image
    enhanced_img_array = model.predict(img_array)

    # Convert the output back to an image format
    # Adjust post-processing if needed
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
        
        # Save the enhanced image to a temporary file for download
        #enhanced_image.save("enhanced_image.png")
        with open("enhanced_image.png", "rb") as file:
            btn = st.download_button(
                label="Download Enhanced Image",
                data=file,
                file_name="enhanced_image.png",
                mime="image/png"
            )
