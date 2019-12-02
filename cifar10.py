# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 09:51:02 2019

@author: STU24
"""
## 12/2 ##
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#from keras.datasets import cifar10 으로도 불려들일 수 있다. 
X_train, X_test, Y_train, Y_test = np.load("C:\\WorkSpace\\cifar10_data.npy",
                                           allow_pickle = True)

X_train.shape
X_test.shape
Y_train.shape
Y_test.shape
plt.figure(figsize=(10,10))
labels=["airplane","automobile","bird","cat","deer","dog",
        "frog","horse","ship","truck"]
for i in range(0,40):
    img = Image.fromarray(X_train[i])
    plt.subplot(5,8,i+1)
    plt.title(labels[Y_train[i][0]])
    plt.imshow(img)
    
plt.show()

#
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

#scaling을 해 보자 
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255

#Y_train을 keras의 함수를 이용해서 one hot encoding으로 바꾸어 보자
Y_train.shape
Y_train = keras.utils.to_categorical(Y_train,10)
Y_train
Y_test = keras.utils.to_categorical(Y_test, 10)
Y_test.shape

#model을 생성하자 
model = Sequential()
model.add(Conv2D(32,#filters
                 (3,3),#kernel_size
                 padding = 'same',
                 input_shape = (32,32,3)))#32*32 RGB 
model.summary()
model.add(Activation('relu'))

model.add(Conv2D(32, (3,3)))
model.summary()
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))#max pooling 작업
model.summary()
model.add(Dropout(0,25))#25%의 weight, bias값을 0으로 채운다.

model.add(Conv2D(64, (3,3), padding = 'same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))#max pooling 작업
model.add(Dropout(0,25))
model.summary()

#펼치기
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))#분류할 갯수
model.add(Activation('softmax'))

model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

hist = model.fit(X_train, Y_train,batch_size = 32,
          epochs = 10, verbose = 1,
          validation_data = (X_test, Y_test))

model

import glob
img = Image.open("C:\\WorkSpace\\car test 1.jpg")

files = glob.glob("C:\\WorkSpace\\car test 1.jpg")
X = []
for i, f in enumerate(files):
    img1 = Image.open(f)
    img1 = img1.convert("RGB")
    img1 = img1.resize([32,32])
    data = np.asarray(img1)
    X.append(data)
    
X = np.array(X)
X = X.astype('float32')/255

model.predict(X, batch_size = 32)

r = model.predict(X, batch_size = 32)
res = r[0]

for i, acc in enumerate(res):
    print(labels[i], '=', int(acc*100))
    
print("예측결과 : ", labels[res.argmax()])

def pre(indata):
    X = []
    for i, f in enumerate(indata):
        img = Image.open(f)
        img = img.convert("RGB")
        img = img.resize([32,32])
        data = np.asarray(img)
        X.append(data)
    
    X = np.array(X)
    X = X.astype('float32')/255
    
    r = model.predict(X, batch_size = 32)
    res = r[0]
    
    for i, acc in enumerate(res):
        print(labels[i], '=', int(acc*100))
    
    print("예측결과 : ", labels[res.argmax()])
    
dog = glob.glob("C:\\WorkSpace\\dog test 1.jpg")
pre(dog)
dog = glob.glob("C:\\WorkSpace\\dog test 2.jpg")
pre(dog)
dog = glob.glob("C:\\WorkSpace\\dog test 3.jpg")
pre(dog)
