import tensorflow as tf
from keras.layers import Dense, Flatten, Lambda, MaxPooling2D, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential
from keras.optimizers import Adam

tf.python.control_flow_ops = tf

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def data_augment(image, steering_angle):
    """
    Randomly translate the image horizontally and vertically

    Then randomly flip the image, and its corresponding steering angle

    """

    if np.random.rand() < 0.9:

        x = 100 * (np.random.rand() - 0.5)
        y = 10 * (np.random.rand() - 0.5)

        steering_angle += x * 0.002
        m = np.float32([[1, 0, x], [0, 1, y]])
        rows, cols = image.shape[:2]
        image = cv2.warpAffine(image, m, (cols, rows))

    if np.random.rand() < 0.5:
        image = np.fliplr(image)
        steering_angle *= -1

    return image, steering_angle

def data_path(batch_size=64):
    """
    Parse CSV training file and generate image file paths
    """

    raw_data = pd.read_csv('udacity_data/driving_log.csv')

    indices = np.random.randint(0, len(raw_data), batch_size)

    paths = []
    for index in indices:
        camera = np.random.choice(['center', 'left', 'right'])
        if camera == 'left':
            img = raw_data.iloc[index]['left'].strip()
            angle = raw_data.iloc[index]['steering'] + 0.2
            paths.append((img, angle))

        elif camera == 'center':
            img = raw_data.iloc[index]['center'].strip()
            angle = raw_data.iloc[index]['steering']
            paths.append((img, angle))
        else:
            img = raw_data.iloc[index]['right'].strip()
            angle = raw_data.iloc[index]['steering'] - 0.2
            paths.append((img, angle))

    return paths

def batch_generator(batch_size=64):
    """
    Generate training image based on image file paths and associated steering angles
    """

    while True:
        X = []
        y = []
        images = data_path(batch_size)

        for img_path, angle in images:

            raw_image = plt.imread('udacity_data/' + img_path)
            raw_angle = angle

            img, ang = data_augment(raw_image, raw_angle)

            X.append(img)
            y.append(ang)

        yield np.array(X), np.array(y)

# Model architecture is subject to NVIDIA model
model = Sequential()

model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((50,20), (0,0))))

# Three 5x5 Convolutional and maxpooling layers w/ feature depth (24, 36, 48), 2x2 stride
model.add(Convolution2D(24, 5, 5, border_mode='same', activation='relu', subsample=(2, 2)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(36, 5, 5, border_mode='same', activation='relu', subsample=(2, 2)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(48, 5, 5, border_mode='same', activation='relu', subsample=(2, 2)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

# Two 3x3 convolutional and maxpooling layers w/ feature depth (64, 64), 1x1 stride
model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu', subsample=(1, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu', subsample=(1, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

# Flatten model
model.add(Flatten())

# Four fully connected layers w/ depth (1164, 100, 50, 10) relu activation
model.add(Dense(1164, activation='relu'))

model.add(Dense(100, activation='relu'))

model.add(Dense(50, activation='relu'))

model.add(Dense(10, activation='relu'))

# Fully connect output layer
model.add(Dense(1))

model.summary()

model.compile(optimizer=Adam(1e-4), loss="mse", )

history_object = model.fit_generator(batch_generator(),
                              samples_per_epoch=20032,
                              nb_epoch=2,
                              validation_data=batch_generator(),
                              nb_val_samples=3000,
                              verbose=1)

# Save model training result
model.save('model.h5')

# Print the keys contained in the history object
print(history_object.history.keys())
# Plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean square error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
