from PIL import Image
import streamlit as st
import io
import base64

def compress_image(input_image, quality=85):
    img_io = io.BytesIO()
    input_image.save(img_io, 'JPEG', quality=quality)
    img_io.seek(0)
    return Image.open(img_io)

def main():
    st.title("Image Compressor")

    # 上传图像
    uploaded_file = st.file_uploader("选择要压缩的图像", type=["jpg", "jpeg"])

    if uploaded_file is not None:
        original_image = Image.open(uploaded_file)
        st.image(original_image, caption="原始图像", use_column_width=True)

        # 获取压缩质量
        quality = st.slider("选择压缩质量（0-100）", 0, 100, 85)

        if st.button("压缩图像"):
            # 压缩图像
            compressed_image = compress_image(original_image, quality)
            st.image(compressed_image, caption="压缩后的图像", use_column_width=True)

            # 显示图像大小
            #original_size_kb = len(uploaded_file.getvalue()) / 1024
            #compressed_size_kb = len(compressed_image.tobytes()) / 1024
            #st.write(f"原始图像大小: {original_size_kb:.2f} KB")
            #st.write(f"压缩后的图像大小: {compressed_size_kb:.2f} KB")


            # 提供下载按钮
            compressed_image_bytes = io.BytesIO()
            compressed_image.save(compressed_image_bytes, format="JPEG")
            compressed_image_bytes = compressed_image_bytes.getvalue()

            download_link = f'<a href="data:application/octet-stream;base64,{base64.b64encode(compressed_image_bytes).decode()}" download="compressed_image.jpg">点击此处下载压缩后的图像</a>'
            st.markdown(download_link, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
