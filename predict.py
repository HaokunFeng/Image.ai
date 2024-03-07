import tensorflow as tf
from keras.models import load_model
import matplotlib.pyplot as plt
from keras.utils import load_img, img_to_array
import numpy as np

# 加载训练好的模型
loaded_model = tf.keras.models.load_model('models/denoising_model.h5')

# 替换为你的图像路径
image_path = 'asset/0010_NOISY_SRGB_010.PNG'

# 加载图像并进行预处理
img = load_img(image_path)
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# 使用模型进行降噪
denoised_img = loaded_model.predict(img_array)

# 可视化结果
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(denoised_img[0])
plt.title('Denoised Image')
plt.axis('off')

plt.show()