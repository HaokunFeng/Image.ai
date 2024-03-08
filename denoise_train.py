import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
import matplotlib.pyplot as plt
import numpy as np

# 加载 CIFAR-10 数据集
(x_train, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()

# 数据预处理
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 构建带有噪声的数据集
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * tf.random.normal(shape=x_train.shape)
x_test_noisy = x_test + noise_factor * tf.random.normal(shape=x_test.shape)

# Clip 超过 0 和 1 的值
x_train_noisy = tf.clip_by_value(x_train_noisy, 0., 1.)
x_test_noisy = tf.clip_by_value(x_test_noisy, 0., 1.)

# 划分数据集
split_ratio = 0.8
split_index = int(len(x_train) * split_ratio)

x_train_clean, x_val_clean = x_train[:split_index], x_train[split_index:]
x_train_noisy, x_val_noisy = x_train_noisy[:split_index], x_train_noisy[split_index:]

# 构建图像降噪模型
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(None, None, 3)))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))

model.compile(optimizer='adam', loss='mse')

# 训练模型
history = model.fit(
    x_train_noisy, x_train_clean,
    epochs=10,
    batch_size=128,
    validation_data=(x_val_noisy, x_val_clean)
)

# 可视化训练结果
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 可视化一些样本的原始图像、带噪声的图像和模型生成的图像
num_samples = 5
selected_indices = np.random.choice(len(x_test), num_samples, replace=False)

for idx in selected_indices:
    original_img = x_test[idx]
    noisy_img = x_test_noisy[idx]
    denoised_img = model.predict(np.expand_dims(noisy_img, axis=0))[0]

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(original_img)
    plt.title('Original Image')

    plt.subplot(1, 3, 2)
    plt.imshow(noisy_img)
    plt.title('Noisy Image')

    plt.subplot(1, 3, 3)
    plt.imshow(denoised_img)
    plt.title('Denoised Image')

    plt.show()

# 保存模型
model.save('models/denoising_model.h5')