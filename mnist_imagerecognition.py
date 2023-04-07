# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
# Dependencies to Visualize the model
# %matplotlib inline
from IPython.display import Image, SVG
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(0)

# Filepaths, numpy, and Tensorflow
import os
import numpy as np
import tensorflow as tf

# Sklearn scaling
from sklearn.preprocessing import MinMaxScaler

# Keras specific dependencies
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import mnist

#Loading and Preprocessing our Data
#Load the MNIST Handwriting Dataset from Keras
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print("Training Data Info")
print("Training Data Shape:", X_train.shape)
print("Training Data Labels Shape:", y_train.shape)
print("Training Data Shape:", X_test.shape)
print("Training Data Labels Shape:", y_test.shape)

#plot first digit
plt.imshow(X_train[0], cmap=plt.cm.Greys)

# Plot the first image from the dataset
plt.imshow(X_train[0,:,:], cmap=plt.cm.Greys)

# image is an array of pixels ranging from 0 to 255
X_train[0, :, :]

# flatten our image of 28x28 pixels to a 1D array of 784 pixels
# print(28*28)
# inputs=[[1,2,3,.........784], 
#         [1,2,3,4,.......784], 
#         ]
ndims = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], ndims)
X_test = X_test.reshape(X_test.shape[0], ndims)
print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

#Scaling and normalization
#Sklearn's MinMaxScaler to normalize our training data between 0 and 1

# new Normalizer
# normalizer.fit
# normalizer.transform

scaler = MinMaxScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

X_train

# Alternative:
# X_train = X_train.astype("float32")
# X_test = X_test.astype("float32")
# X_train /= 255.0
# X_test /= 255.0

# one-hot encode our integer labels using the to_categorical helper function
# Training and Testing labels are integer encoded from 0 to 9
y_train[:20]

# convert our target labels (expected values) to categorical data
num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
# Original label of `5` is one-hot encoded as `0000010000`
y_train[0]

# Deep Multi-Layer Perceptron model with 2 hidden layers
# Create an empty sequential model
# new model 
# model.fit
# model.predict

# new model
# model add layer
model=Sequential()

# first layer where the input dimensions are the 784 pixel values
# can also choose our activation function. `relu` is a common
pixels=28
model.add(Dense(100, activation='relu', input_dim=pixels**2))

# second hidden layer
model.add(Dense(100, activation='relu'))

# final output layer where the number of nodes 
# final output layer uses a softmax activation function for logistic regression
# corresponds to the number of y labels
model.add(Dense(num_classes, activation='softmax'))

# model summary
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit (train) the model
model.fit(X_train, y_train, epochs=10, shuffle=True, verbose=2)

# Save the model
model.save('mnist_trained.h5')

# Load the model
from tensorflow.keras.models import load_model
loaded_model=load_model('mnist_trained.h5')

# Evaluatec the model using the training data 
model_loss, model_accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")

# test with one data point
test = np.expand_dims(X_train[0], axis=0)
test.shape
test

plt.imshow(scaler.inverse_transform(test).reshape(28, 28), cmap=plt.cm.Greys)

# Predicting: The result should be 0000010000000 for a 5
model.predict(test).round()

# Test with one data point
test = np.expand_dims(X_train[2], axis=0)
test.shape

plt.imshow(scaler.inverse_transform(test).reshape(28, 28), cmap=plt.cm.Greys)

# Prediction
print(f"One-Hot-Encoded Prediction: {model.predict(test).round()}")
print(f"Predicted class: {model.predict_classes(test)}")

"""# Import a Custom Image"""

filepath = "/content/test2.png"
filepath='/content/test2.png'

# Import the image using the `load_img` function in keras preprocessing
from tensorflow.keras.preprocessing import image
image_size=(28, 28)
sample=image.load_img(filepath, target_size=image_size, color_mode='grayscale')
sample

# Convert the image to a numpy array 
from tensorflow.keras.preprocessing.image import img_to_array
image=img_to_array(sample)
image.shape

# Scale the image
image=image/255
image

# Flatten into a 1x28*28 array 
image=image.flatten().reshape(-1, pixels**2)
image.shape

plt.imshow(image.reshape(28, 28), cmap=plt.cm.Greys)

# Invert the pixel values to match the original data
image=1-image
plt.imshow(image.reshape(28, 28), cmap=plt.cm.Greys)

# Make predictions
model.predict_classes(image)

