import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
import matplotlib.pyplot as plt
import numpy as np

# load CIFAR-10 dataset
(x_train, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()

# preprocess the data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# add random noise to the images
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * tf.random.normal(shape=x_train.shape)
x_test_noisy = x_test + noise_factor * tf.random.normal(shape=x_test.shape)

# clip
x_train_noisy = tf.clip_by_value(x_train_noisy, 0., 1.)
x_test_noisy = tf.clip_by_value(x_test_noisy, 0., 1.)

# split the data into training and validation sets
split_ratio = 0.8
split_index = int(len(x_train) * split_ratio)

x_train_clean, x_val_clean = x_train[:split_index], x_train[split_index:]
x_train_noisy, x_val_noisy = x_train_noisy[:split_index], x_train_noisy[split_index:]

# build the model
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

# train the model
history = model.fit(
    x_train_noisy, x_train_clean,
    epochs=10,
    batch_size=128,
    validation_data=(x_val_noisy, x_val_clean)
)

# visualize the training process
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# visualize all samples on the same figure
num_samples = 10
selected_indices = np.random.choice(len(x_test), num_samples, replace=False)

plt.figure(figsize=(15, 4))

for i, idx in enumerate(selected_indices):
    original_img = x_test[idx]
    noisy_img = x_test_noisy[idx]
    denoised_img = model.predict(np.expand_dims(noisy_img, axis=0))[0]

    # Original Image
    plt.subplot(3, num_samples, i + 1)
    plt.imshow(original_img)
    plt.title(f'Sample {i + 1}\nOriginal Image')
    plt.axis('on')

    # Noisy Image
    plt.subplot(3, num_samples, num_samples + i + 1)
    plt.imshow(noisy_img)
    plt.title('Noisy Image')
    plt.axis('on')

    # Denoised Image
    plt.subplot(3, num_samples, 2 * num_samples + i + 1)
    plt.imshow(denoised_img)
    plt.title('Denoised Image')
    plt.axis('on')

plt.tight_layout()
plt.show()


model.save('models/denoising_model.h5')