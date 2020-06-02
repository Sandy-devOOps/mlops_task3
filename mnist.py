from keras.datasets import mnist
dataset = mnist.load_data('mymnist.db')
train , test = dataset
X_train , y_train = train
X_test , y_test = test
img1 = X_train[7]
import cv2
img1_label = y_train[7]
X_train_1d = X_train.reshape(-1 , 28*28)
X_test_1d = X_test.reshape(-1 , 28*28)
X_train = X_train_1d.astype('float32')
X_test = X_test_1d.astype('float32')
X_train.shape
from keras.utils.np_utils import to_categorical
y_train_cat = to_categorical(y_train)
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(units=512, input_dim=28*28, activation='relu'))
model.add(Dense(units=256, activation='relu'))
i=0
for i in range(i):
    model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=10, activation='softmax'))
from keras.optimizers import RMSprop
model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', 
             metrics=['accuracy']
             )
h = model.fit(X_train, y_train_cat, epochs=20)
model.predict(X_test)
p=h.history['accuracy']
with open('file.txt','w') as f:
    f.write(str(p[7]))
model.save('mnist.py')