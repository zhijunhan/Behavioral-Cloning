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
### Model Architecture

The project model was suggested from Udacity instructions, it is based on a well known self-driving car model, [NVIDIA model](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). The following picture should provide a description of its architecture.

<img width="474" alt="screen shot 2018-02-03 at 01 25 45" src="https://user-images.githubusercontent.com/21351949/35763769-426fd35c-0881-11e8-92a8-200bfbc77f71.png">

Note that the input data shape is `(160, 320, 3)`, cropping function is applied.

The model architecture used in this project specifically has five convolutional and maxpooling layers with feature depth of (24, 36, 48) and three fully connected layers with depth of (1164, 100, 50) and one output layer, all layers have relu activation method.

Image normalization was implemented using a Keras Lambda function, and image data are cropped using a Keras Cropping function. The paper does not address any activation method or model overfitting mitigation, so a typical relu activation functions on each fully connected and convolutional layer are used.

Here is the model architecture schematic:

Layer (type)                     Output Shape          Param #     Connected to                     
____________________________________________________________________________________________________
lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_1[0][0]             

cropping2d_1 (Cropping2D)        (None, 90, 320, 3)    0           lambda_1[0][0]                   

convolution2d_1 (Convolution2D)  (None, 45, 160, 24)   1824        cropping2d_1[0][0]               

maxpooling2d_1 (MaxPooling2D)    (None, 44, 159, 24)   0           convolution2d_1[0][0]            

convolution2d_2 (Convolution2D)  (None, 22, 80, 36)    21636       maxpooling2d_1[0][0]             

maxpooling2d_2 (MaxPooling2D)    (None, 21, 79, 36)    0           convolution2d_2[0][0]            

convolution2d_3 (Convolution2D)  (None, 11, 40, 48)    43248       maxpooling2d_2[0][0]             

maxpooling2d_3 (MaxPooling2D)    (None, 10, 39, 48)    0           convolution2d_3[0][0]            

convolution2d_4 (Convolution2D)  (None, 10, 39, 64)    27712       maxpooling2d_3[0][0]             

maxpooling2d_4 (MaxPooling2D)    (None, 9, 38, 64)     0           convolution2d_4[0][0]            

convolution2d_5 (Convolution2D)  (None, 9, 38, 64)     36928       maxpooling2d_4[0][0]             

maxpooling2d_5 (MaxPooling2D)    (None, 8, 37, 64)     0           convolution2d_5[0][0]            

flatten_1 (Flatten)              (None, 18944)         0           maxpooling2d_5[0][0]             

dense_1 (Dense)                  (None, 1164)          22051980    flatten_1[0][0]                  

dense_2 (Dense)                  (None, 100)           116500      dense_1[0][0]                    

dense_3 (Dense)                  (None, 50)            5050        dense_2[0][0]   

dense_4 (Dense)                  (None, 10)            510         dense_3[0][0]                    

dense_5 (Dense)                  (None, 1)             11          dense_4[0][0]                    

### Data Collection

The Car Simulator is used to acquire road image and steering angle data. For each given sampling time, a set of left, center and right camera image data would be collected.

### Data Processing

The following steps summarize the data processing:

#### 1. Random Flip

Randomly flip image and associate steering angles at a chance of 50%

#### 2. Data Augmentation

Apply image transformation function, the function warpAffine transforms the source image using the specified matrix using the following math:

```
dst(x,y) = src(M11 * x + M12 * y + M13, M21 * x + M22 * y + M23)
```

#### 3. Data Cropping

A Keras Cropping function was used to crop image by removing 50 and 20 pixels from top and bottom.

### Training

From the training efficiency point of view, a Keras fit_generator API was used to train our model with a batch size of 64. Adam optimizer with a learning rate of 1e-4 was used for back propagation. The optimization score function (loss function) is Mean Squared Error. And training epoch is set to 2.

## Conclusions and Results
---
The trained car is capable of driving along the middle of the road on the training track endlessly. Following plot shows the training and validation loss for each epoch of the model.

![loss](https://user-images.githubusercontent.com/21351949/35763756-fe7ca076-0880-11e8-94de-f4edc0da7374.png)

It is very clear that this self-driving car is a very basic example just for demonstration. However, it should present a good idea of what the whole process has been capable of, give the fact that the training is so limited that the data are acquired using only on a car driving simulator.
