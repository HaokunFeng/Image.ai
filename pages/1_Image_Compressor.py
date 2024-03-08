from PIL import Image
import streamlit as st
import io
import base64

st.set_page_config(
    page_title="Image Compressor",
    page_icon="🗜️",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)
st.title("🗜️ Image Compressor")
st.caption("Upload image and compress the size of it without losing quality.")
st.markdown("---")

def compress_image(input_image, quality=85):
    img_io = io.BytesIO()
    input_image.save(img_io, 'JPEG', quality=quality)
    img_io.seek(0)
    return Image.open(img_io)

def main():
    st.title("Image Compressor")

    uploaded_file = st.file_uploader("Select your image", type=["jpg", "jpeg"])

    if uploaded_file is not None:
        original_image = Image.open(uploaded_file)
        st.image(original_image, caption="Original Image", use_column_width=True)


        quality = st.slider("Select Compression Quality(0-100)", 0, 100, 85)

        if st.button("Compress Image"):
            compressed_image = compress_image(original_image, quality)
            st.image(compressed_image, caption="Compressed image", use_column_width=True)

            #original_size_kb = len(uploaded_file.getvalue()) / 1024
            #compressed_size_kb = len(compressed_image.tobytes()) / 1024
            #st.write(f"Original Size: {original_size_kb:.2f} KB")
            #st.write(f"Compressed Size: {compressed_size_kb:.2f} KB")


            compressed_image_bytes = io.BytesIO()
            compressed_image.save(compressed_image_bytes, format="JPEG")
            compressed_image_bytes = compressed_image_bytes.getvalue()

            download_link = f'<a href="data:application/octet-stream;base64,{base64.b64encode(compressed_image_bytes).decode()}" download="compressed_image.jpg">download compressed image</a>'
            st.markdown(download_link, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
