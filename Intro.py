import streamlit as st

st.set_page_config(
    page_title="ImageAI",
    page_icon="🚩",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)
st.title("📷 Image AI")
st.caption("A collection of image processing applications using deep learning models.")

st.markdown("#### Greetings 👋")
st.markdown(
    "This web app is a collection of image processing applications using deep learning models. "
    "You can use the sidebar to navigate between different applications."
)

st.markdown("---")
st.markdown("####  Features 🎯")
st.markdown(
    "1. **Image Compression**: Compress the size of an image without losing quality."
)
st.markdown(
    "2. **Image Denoising**: Remove noise from images using a our pre-trained model."
)
st.markdown(
    "3. **Image Enhancement**: Enhance the quality of images using a our pre-trained model."
)

st.markdown("---")
st.markdown("#### Privacy 📂")
st.markdown(
    "At ImageAI, your privacy is our top priority. To protect your personal information, our system only uses your files data when needed, and will not save the data, ensuring complete privacy and anonymity. This means you can use ImageAI with peace of mind, knowing that your data is always safe and secure."
)