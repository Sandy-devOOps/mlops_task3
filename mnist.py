#!/usr/bin/env python
# coding: utf-8

from keras.datasets import mnist
(X_train, y_train), (X_test, y_test)  = mnist.load_data()
img_rows = X_train[0].shape[0]
img_cols = X_train[1].shape[0]
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
X_train.shape
input_shape = (img_rows, img_cols, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
from keras.models import Model,Sequential
from keras.layers import Flatten,Dense
from keras.layers import Convolution2D, MaxPooling2D,ZeroPadding2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.utils import np_utils
model = Sequential()
model.add(Convolution2D(32, (3, 3), activation='relu',input_shape=input_shape))
model.add(MaxPooling2D( pool_size=(2, 2)))
model.add(Flatten())

model.add(Dense(units=512,input_dim=28*28,  activation='relu'))
model.add(Dense(units=256, activation='relu'))
model.add(Dense(units=128, activation='relu'))

model.add(Dense(units=10, activation='softmax'))
from keras.optimizers import RMSprop
model.compile(optimizer=RMSprop(),loss='categorical_crossentropy', 
             metrics=['accuracy']
             )
  
print(model.summary())
batch_size = 128
epochs = 3

history = model.fit(X_train, y_train,
          batch_size=batch_size,         
          epochs=epochs,
          validation_data=(X_test, y_test),
          shuffle=True)
model.predict(X_test)

scores = model.evaluate(x_test, y_test, verbose=1)
accuracy_score=scores[1]
f=open("output.txt","w")
f.write(str(100*accuracy_score))
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
