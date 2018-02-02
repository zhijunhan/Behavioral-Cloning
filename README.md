# Behaviorial Cloning Project

Overview
---
The objective of this project is to clone a human driving behavior by using Deep Neural Network implemented in Keras and Tensorflow. A Car Simulator is used to generate training images and respective steering angles. Then those data will be used to train the neural network. Trained model will be tested on training track.

## Dependencies
---
Following libraries and utilities are used for this project:

- [Keras](https://keras.io/)
- [Pandas](http://pandas.pydata.org/)
- [OpenCV](http://opencv.org/)
- [Matplotlib](http://matplotlib.org/)
- [NumPy](http://www.numpy.org/)
- [TensorFlow](http://tensorflow.org)

## Project Implementation
---
### MOdel Architecture

The project model was suggested from Udacity instructions, it is based on a well known self-driving car model, [NVIDIA model](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). The following picture should provide a description of its architecture.

The model architecture used in this project specifically has five convolutional and maxpooling layers with feature depth of (24, 36, 48) and three fully connected layers with depth of (1164, 100, 50) and one output layer, all layers have relu activation method. Dropout was used for each of the fully connected layers.

Image normalization was implemented using a Keras Lambda function, and image data are cropped using a Keras Cropping function. Image data are converted from RGV to YUV color space, as suggested in the paper. The paper does not address any activation method or model overfitting mitigation, so a typical relu activation functions on each fully connected and convolutional layer are used. The dropout function with a keep probability of 0.5 between each of fully connected layer was used. 

### Data Collection

The Car Simulator is used to acquire road image and steering angle data. For each given sampling time, a set of left, center and right camera image data would be collected.

### Data Processing

The following steps summarize the data processing:

#### Random Flip

Randomly flip image and respective angle data when steering angle magnitude is larger than 0.25

#### Adjust Brightness

Image brightness was brought up.

#### Data Augmentation

For each sampling step, the set of center, left and right camera data are both processed.

#### Color Adjust

Image converted from RGB to YUV color space, as suggested by NVIDIA paper.

#### Data Cropping

A Keras Cropping function was used to crop image by removing 50 and 20 pixels from top and bottom.

### Training

From the training efficiency point of view, a Keras fit_generator API was used to train our model with a batch size of 64. Adam optimizer with a learning rate of 1e-4 was used for back propagation.

## Results
---

