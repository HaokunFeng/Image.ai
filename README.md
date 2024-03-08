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

### Results
Due to the manual addition of image noise in the training set, the model trained by this method has weak generalization ability and can perform well on artificial noise. However, for natural noise, such as noise caused by lighting, color, camera errors, etc., the noise reduction effect is slightly worse.

![Test Results](https://github.com/HaokunFeng/Image.ai/blob/main/assets/denoise_2.png)

### Samples
![Sample1](https://github.com/HaokunFeng/Image.ai/blob/main/assets/sample_1.jpg)
1[Sample2](https://github.com/HaokunFeng/Image.ai/blob/main/assets/sample_1_dnoised.jpg)



## Image Super-Resolution Using Convolutional Neural Networks

This project demonstrates the application of Convolutional Neural Networks (CNNs) for enhancing the resolution of images, a process commonly known as Image Super-Resolution (ISR). We utilize the DIV2K dataset, a popular benchmark for super-resolution, to train and evaluate our model.

### Dataset

The DIV2K dataset is a high-quality resource for training and benchmarking super-resolution algorithms. It comprises 1,000 images, split into:

- 800 training images
- 100 validation images
- 100 test images

Each high-resolution (HR) image in the dataset has a corresponding low-resolution (LR) counterpart, generated through bicubic downsampling. In this project, we focus on the x4 downsampling factor, which means the LR images are 1/4th the height and width of the HR images.

More details about the DIV2K dataset can be found [here](https://data.vision.ee.ethz.ch/cvl/DIV2K/).

### Methodology

#### Model Architecture

We implemented a simple CNN model for this task, which consists of the following layers:

1. An input layer that accepts LR images.
2. Three convolutional layers with ReLU activation to extract features and upscale the image.
3. An output layer that produces the super-resolved image.

#### Preprocessing

The preprocessing steps include:
- Normalizing the pixel values of both LR and HR images to the [0, 1] range.
- Randomly cropping the images to 64x64 patches during training to facilitate batch processing.
- Using data augmentation techniques such as flipping and rotation to enhance the diversity of the training data.

#### Training

The model was trained using the following settings:
- Loss function: Mean Squared Error (MSE)
- Optimizer: Adam
- Batch size: 16
- Number of epochs: 10
  
#### Loss Curves

The following graph illustrates the training and validation loss over the epochs:

<img width="575" alt="image" src="https://github.com/hengjunzhang/513-final/assets/146139584/217188b7-5fcd-41e2-a174-17f9ef5bc93b">


As shown in the graph, both losses decreased over time and the validation loss tracked closely with the training loss, which is a good indication of the model's performance.


### Results

The trained model demonstrates a notable improvement in the quality of upsampled images compared to bicubic interpolation. However, for optimal performance and results competitive with state-of-the-art models, further tuning and a more complex architecture may be necessary.

#### Quantitative Results

- Average Peak Signal-to-Noise Ratio (PSNR) on validation set: XX dB
- Structural Similarity Index (SSIM) on validation set: XX

#### Qualitative Results

<<img width="791" alt="image" src="https://github.com/hengjunzhang/513-final/assets/146139584/82c05585-ce69-435d-8c1c-3246c39ac417">
>


### Conclusion and Future Work

This project showcases the potential of CNNs in enhancing image resolution. Future work could explore more sophisticated models such as ESRGAN or SRGAN for improved performance. Additionally, experimenting with different loss functions like perceptual loss could yield more visually pleasing results.

## How to Use

To train the model on your dataset:
1. Prepare your dataset in the required format.
2. Adjust the model architecture and training parameters as needed.
3. Run the training script: `python train.py`

To enhance images using the trained model:
1. Load your model: `model = load_model('path_to_your_model.h5')`
2. Preprocess your low-resolution image as described in the preprocessing section.
3. Use the model to predict the super-resolved image: `sr_img = model.predict(lr_img)`
4. Display or save the super-resolved image.

## Requirements

- TensorFlow 2.x
- TensorFlow Datasets
- Matplotlib
- NumPy
- Scipy
- Keras
- Streamlit
