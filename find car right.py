# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 13:59:58 2019

@author: STU24
"""

#우측만 해 보자#

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

caltech_dir = "C:/WorkSpace/screen shots right test"
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

FullCount = 0
FreeCount = 0
for i in Y:
    if i.argmax() == 0:
        FullCount += 1
    else:
        FreeCount += 1
print(FullCount)
print(FreeCount)    

##### 데이터를 생성하고 싶은 경우에만 #####
data_datagen = ImageDataGenerator(
        rotation_range = 0, #이미지 회전 범위 (각도) (0 ~ 180)
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        #그림을 수평 또는 수직으로 랜덤하게 평행 이동시키는 범위
        #(원본 가로, 세로 길이에 대한 비율값)
        shear_range = 0.1, #y축 방향으로 각도를 증가시켜 이미지를 변형시킨다.
        zoom_range = 0.1, #확대, 축소 범위: 
        #[lower, upper] = [1-zoom_range, 1+zoom_range]
        horizontal_flip = False, 
        #True 로 설정할 경우 50%확률로 이미지를 수평으로 뒤집는다.        
        fill_mode = 'nearest' 
        # 이미지를 회전, 이동하거나 축소할 때 생기는 공간을 채우는 방식
        #'constant','nearest','reflect','wrap' 4가지 방법이 있다.
        # 'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
        # 'nearest': aaaaaaaa|abcd|dddddddd
        # 'reflect': abcddcba|abcd|dcbaabcd
        # 'wrap': abcdabcd|abcd|abcdabcd
        #rescale : 원본 영상은 0 ~ 255 사이의 값으로 (RGB계수)구성되어 있다.
        #이 같은 입력값은 모델을 효과적으로 학습시키기에 너무 높다. 
        #그래서 이를 1/255로 스케일링하여 0 ~ 1 범위로 변환시켜준다. 
        )

for idx, cat in enumerate(["Full"]):
    
    image_dir = caltech_dir + "/" + cat
    files = glob.glob(image_dir + "\\*.jpg")
    
    #이제 차례대로 이미지 파일을 불러와서 변형한 후에 저장하자 
    for f in files:
        img = load_img(f)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        i = 0
        for batch in data_datagen.flow(x,
                    save_to_dir = image_dir,
                    save_prefix = '20193333', save_format = 'jpg'):
            i += 1
            if i > 2:
                break

########
   
X_train, X_test, Y_train,Y_test = train_test_split(X,Y,test_size=0.2)
int(len(X_train) /4)

base_model = MobileNetV2(input_shape=(224,224,3),
                         weights='imagenet',include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x) 
output = Dense(2, activation="softmax")(x)

model = Model(inputs = base_model.input, outputs=output)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc'])

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

hist = model.fit(X_train,Y_train,batch_size= 32,
                 epochs=5,verbose=1,
                 validation_data=(X_test,Y_test),
                 callbacks=[
                         ModelCheckpoint('modelR.h5',
                                         monitor='val_acc',
                                         save_best_only=True,
                                         verbose=1)
                         ]
                 )

#################################################
#Create New Model
#################################################

model = load_model('modelR.h5')

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
        'Free/20191204221730_1.jpg'),
                target_size=(224,224,3)))

test_input = preprocess_input(np.expand_dims(test_img.copy(),axis=0))
test_input.shape
#(1,224,224,3)

pred = model.predict(test_input)

plt.figure(figsize=(8,8))
plt.title('%.2f%% Free'%(pred[0][1]*100))
plt.imshow(test_img.astype(np.uint8))


pred2 = model.predict(X_test, batch_size = 32)
pred2.argmax(axis = 1)
Y_test.argmax(axis = 1)

(pred2.argmax(axis = 1) == Y_test.argmax(axis = 1)).sum() / len(Y_test)

def test_data_show(i):
    print(categories[pred2[i].argmax()])
    plt.imshow(X_test[i])

test_data_show(2)
test_data_show(12)
test_data_show(44)

def train_data_show(i):
    test_train_data = X_train[i].reshape(1,224,224,3)
    predict_sample = model.predict(test_train_data)
    print(categories[predict_sample.argmax()])
    plt.imshow(X_train[i])
    
    
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
caltech_dir = "C:/WorkSpace/car right test"
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

pred_car = model.predict(X_car, batch_size = 32)
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

def DrawActivationMapShow(i):
    input_test = X_car[i].reshape(1,224,224,3)
    last_conv_output, pred = new_model.predict(input_test)
    last_conv_output = np.squeeze(last_conv_output) #(7,7,1280)
    feature_activation_maps = scipy.ndimage.zoom(
            last_conv_output,(32,32,1),order=1)
    pred_class = np.argmax(pred)
    predicted_class_weights = last_weight[:, pred_class]
    
    final_output = np.dot(feature_activation_maps.reshape((224*224,1280)),
                      predicted_class_weights).reshape((224,224))
    
    #plt.imshow(final_output,cmap='jet')
    print('정답 : ',categories[Y_car[i].argmax()])
    print("예측 : ",categories[pred_class])
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
    
#틀린것만 보기 
for j in range(len(X_car)):
        if pred_car[j].argmax() != Y_car[j].argmax():
                DrawActivationMapShow(j)

#맞는것만 보기 
for j in range(len(X_car)):
        if pred_car[j].argmax() == Y_car[j].argmax():
                car_test_show(j)