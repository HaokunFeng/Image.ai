# Image.ai

This web app is a collection of image processing applications using deep learning models. You can use the sidebar to navigate between different applications.

## Features of the App 🎯
- **Image Compression**: Compress the size of an image without losing quality.
- **Image Denoising**: Remove noise from images using a our pre-trained model.
- **Image Enhancement**: Enhance the quality of images using a our pre-trained model.

## Get Started 🥳
To use this program in your local environment, follow these simple steps:

- Clone this repository.
- ``python -m venv venv``
- ``.\venv\Scripts\Activate`` (Or use a proper command based on your OS to activate the environment)
- ``pip install -r requirements.txt``
- ``streamlit run intro.py`` to run the ImageAI
- To train your own model, use the ``.py`` or ``.ipynb`` files in ``train`` folder


## Image Denoiser
Image Denoiser implements a convolutional autoencoder for image denoising using the CIFAR-10 dataset.

### Dataset
The [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.
- **Images**: The dataset consists of a collection of small, color images.
- **Classes**: There are 10 distinct classes or categories for image classification.
- **Resolution**: Each image is of size 32x32 pixels.

### Training

The model was trained using the following settings:
- Loss function: Mean Squared Error (MSE)
- Optimizer: Adam
- Batch size: 128
- Number of epochs: 10

### Loss Curves
The following graph illustrates the training and validation loss over the epochs:
![Loss Curves](https://github.com/HaokunFeng/Image.ai/blob/main/assets/denoise_1.png)

