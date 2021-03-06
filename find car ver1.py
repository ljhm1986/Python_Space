# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 14:51:25 2019

@author: STU24
"""

import keras
import os, glob
import scipy
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras.layers import GlobalAveragePooling2D,Dense
from keras.models import Model,load_model
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
import tensorflow as tf

keras.__version__

caltech_dir = "C:/WorkSpace/screen shots test"
categories = ["Full","Free"]
nb_class = len(categories)
image_w = 224
image_h = 224
pixels = image_w * image_h * 3
X = []
Y = []
for idx, cat in enumerate(categories):
	label = [0 for i in range(nb_class)]
	label[idx] = 1
	image_dir = caltech_dir+"/"+cat
	files = glob.glob(image_dir+"/*.jpg")
	print(files)
	for i, f in enumerate(files):
		img = Image.open(f)
		img = img.convert("RGB")
		img = img.resize((image_w,image_h))
		data = np.asarray(img)
		X.append(data)
		Y.append(label)
		if i % 10 == 0:
			print(i,"\n",data)
            
X = np.array(X)
Y = np.array(Y)

X_train, X_test, Y_train,Y_test = train_test_split(X,Y,test_size=0.2)


base_model = MobileNetV2(input_shape=(224,224,3),
                         weights='imagenet',include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x) 
output = Dense(2, activation="softmax")(x)

model = Model(inputs = base_model.input, outputs=output)

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])

model.summary()

######################################################
#Switch Layers to be Trainable
######################################################

for layer in model.layers:
    layer.trainable=True
#False의 경우 layer의weight값이 변하지않기에 혹시몰라 True로 지정해주는 것  

###################################################
#Train
###################################################

hist = model.fit(X_train,Y_train,batch_size=32,
                 epochs=10,verbose=1,
                 validation_data=(X_test,Y_test),
                 callbacks=[
                         ModelCheckpoint('model.h5',
                                         monitor='val_acc',
                                         save_best_only=True,
                                         verbose=1)
                         ]
                 )

#################################################
#Create New Model
#################################################

model = load_model('model.h5')

last_weight = model.layers[-1].get_weights()[0]#(1280,2)

new_model = Model(
        inputs=model.input,
        outputs=(
                model.layers[-3].output,
                model.layers[-1].output))

new_model.summary()

test_img = img_to_array(load_img(
        os.path.join(
        caltech_dir,
        'Free/20191204234937_1.jpg'),
                target_size=(224,224,3)))

test_input = preprocess_input(np.expand_dims(test_img.copy(),axis=0))

pred = model.predict(test_input)

plt.figure(figsize=(8,8))
plt.title('%.2f%% Free'%(pred[0][1]*100))
plt.imshow(test_img.astype(np.uint8))


pred2 = model.predict(X_test)
pred2.argmax(axis = 1)
Y_test.argmax(axis = 1)

(pred2.argmax(axis = 1) == Y_test.argmax(axis = 1)).sum() / len(Y_test)

def test_data_show(i):
    print(categories[pred2[i].argmax()])
    plt.imshow(X_test[i])
    plt.show()

test_data_show(2)
test_data_show(12)
test_data_show(44)

def train_data_show(i):
    test_train_data = X_train[i].reshape(1,224,224,3)
    predict_sample = model.predict(test_train_data)
    print(categories[predict_sample.argmax()])
    plt.imshow(X_train[i])
    plt.show()
    
    
train_data_show(5)
train_data_show(33)
train_data_show(44)

test_data1 = X_train[44]

test_input = preprocess_input(np.expand_dims(test_data1.copy(),axis=0))
pred = model.predict(test_input)

plt.figure(figsize=(8,8))
plt.title('%.2f%% Free'%(pred[0][1]*100))
plt.imshow(X_train[44])

###
caltech_dir = "C:/WorkSpace/car test"
categories = ["Full","Free"]
nb_class = len(categories)
image_w = 224
image_h = 224
pixels = image_w * image_h * 3
X_car = []
Y_car = []
for idx, cat in enumerate(categories):
	label = [0 for i in range(nb_class)]
	label[idx] = 1
	image_dir = caltech_dir+"/"+cat
	files = glob.glob(image_dir+"/*.jpg")
	print(files)
	for i, f in enumerate(files):
		img = Image.open(f)
		img = img.convert("RGB")
		img = img.resize((image_w,image_h))
		data = np.asarray(img)
		X_car.append(data)
		Y_car.append(label)
		if i % 10 == 0:
			print(i,"\n",data)
            
X_car = np.array(X_car)
Y_car = np.array(Y_car)

pred_car = model.predict(X_car)
(pred_car.argmax(axis = 1) == Y_car.argmax(axis = 1)).sum() / len(X_car)

def car_test_show(i):
        car_test_data = X_car[i].reshape(1,224,224,3)
        pred_car1 = model.predict(car_test_data)
        print(categories[pred_car1.argmax()])
        plt.imshow(X_car[i])
        plt.show()

car_test_show(0)
car_test_show(2)
car_test_show(4)
car_test_show(5)
car_test_show(6)
car_test_show(7)

#틀린것만 보기 
for j in range(len(X_car)):
        if pred_car[j].argmax() != Y_car[j].argmax():
                car_test_show(j)

#맞는것만 보기 
for j in range(len(X_car)):
        if pred_car[j].argmax() == Y_car[j].argmax():
                car_test_show(j)

Y_FREE = []
for i in Y_car:
    if i.argmax() == 1:
        Y_FREE.append(i)
        
Y_FREE = np.array(Y_FREE)
len(Y_FREE)
len(Y_car)

X_FREE = X_car[24-15:]
len(X_FREE)
pred_car[24-15,:].argmax(axis = 0)

input_test = X_FREE[4].reshape(1,224,224,3)

####################################################
#Draw Activation Map
####################################################

#last_conv_output, pred = new_model.predict(test_input)
last_conv_output, pred = new_model.predict(input_test)

last_conv_output = np.squeeze(last_conv_output) #(7,7,1280)
feature_activation_maps = scipy.ndimage.zoom(last_conv_output,(32,32,1),order=1)#order값 안보임
#(7,7,1280)->(224,224,1280)
#order 값 0(nearest), 1(쌍원보간),3(기본값 cubic)

pred_class = np.argmax(pred)# 0: full, 1: free
predicted_class_weights = last_weight[:, pred_class]#(1280,1)

final_output = np.dot(feature_activation_maps.reshape((224*224,1280)),
                      predicted_class_weights).reshape((224,224))
#(224*224, 1280) dot_product (1280,1) = (224*224,1)

plt.imshow(final_output,cmap='jet')

fig, ax = plt.subplots(nrows=1,ncols=2)
fig.set_size_inches(16,20)

ax[0].imshow(input_test[0])
ax[0].set_title('image')
ax[0].axis('off')

ax[1].imshow(input_test[0],alpha=0.5)
ax[1].imshow(final_output,cmap='jet',alpha=0.5)
ax[1].set_title('class activation map')
ax[1].axis('off')
plt.show()