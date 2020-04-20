# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 00:41:53 2020

@author: ASUS
"""


import matplotlib.pyplot as plt
import imgDataset
import keras as keras
import tensorflow as tf

image_index = 3465


print(imgDataset.x_train[image_index])
#print(imgDataset.y_train[image_index])
plt.imshow(imgDataset.x_train[image_index],cmap='Greys')

plt.show()

print(imgDataset.x_train.shape)

print(imgDataset.x_train.shape[0])

# Reshaping the array to 4-dims so that it can work with the Keras API
imgDataset.x_train =imgDataset.x_train.reshape(imgDataset.x_train.shape[0], 28, 28, 1)
imgDataset.x_test = imgDataset.x_test.reshape(imgDataset.x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
# Making sure that the values are float so that we can get decimal points after division
imgDataset.x_train = imgDataset.x_train.astype('float32')
imgDataset.x_test = imgDataset.x_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
imgDataset.x_train /= 255
imgDataset.x_test /= 255
print('x_train shape:', imgDataset.x_train.shape)
print('Number of images in x_train',imgDataset. x_train.shape[0])
print('Number of images in x_test', imgDataset.x_test.shape[0])
    


from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))


model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(x=imgDataset.x_train,y=imgDataset.y_train, epochs=10)



model.evaluate(imgDataset.x_test, imgDataset.y_test)

image_index = 4444
plt.imshow(imgDataset.x_test[image_index].reshape(28, 28),cmap='Greys')
pred = model.predict(imgDataset.x_test[image_index].reshape(1, 28, 28, 1))
print(pred.argmax())