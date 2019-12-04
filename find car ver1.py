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

caltech_dir = "C:/WorkSpace/SCREENSHOTS"
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

hist = model.fit(X_train,Y_train,batch_size=32,epochs=10,verbose=1,validation_data=(X_test,Y_test),
                 callbacks=[ModelCheckpoint('model.h5',monitor='val_acc',save_best_only=True,verbose=1)])

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
        'Free/20191203220742_1.jpg'),
                target_size=(224,224)))

test_input = preprocess_input(np.expand_dims(test_img.copy(),axis=0))

pred = model.predict(test_input)

plt.figure(figsize=(8,8))
plt.title('%.2f%% Free'%(pred[0][1]*100))
plt.imshow(test_img.astype(np.uint8))

####################################################
#Draw Activation Map
####################################################

last_conv_output, pred = new_model.predict(test_input)

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

ax[0].imshow(test_img.astype(np.uint8))
ax[0].set_title('image')
ax[0].axis('off')

ax[1].imshow(test_img.astype(np.uint8),alpha=0.5)
ax[1].imshow(final_output,cmap='jet',alpha=0.5)
ax[1].set_title('class activation map')
ax[1].axis('off')
plt.show()