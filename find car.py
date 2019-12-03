# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 11:10:26 2019

@author: STU24
"""

##주차공간이 비어있는지 보기 

import os, glob
import scipy
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint

#다운로드한 경로를 입력하고 
BASE_PATH = 'C:\\WorkSpace\\find-a-car-park\\data'

#이미지 파일 경로를 담은 list를 만들자 
full_list = glob.glob(os.path.join(BASE_PATH, 'Full\\*.jpg'))
free_list = glob.glob(os.path.join(BASE_PATH, 'Free\\*.jpg'))

#이미지 중에서 맨 앞에 하나만 살펴보자 
full_img = img_to_array(load_img(full_list[0]), dtype=np.uint8)
free_img = img_to_array(load_img(free_list[0]), dtype=np.uint8)

#이미지 출력 
plt.figure(figsize=(16, 16))
plt.subplot(1, 2, 1)
plt.title(len(full_list))
plt.imshow(full_img)
plt.subplot(1, 2, 2)
plt.title(len(free_list))
plt.imshow(free_img)

#이미지 생성 
train_datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.8, 1.2],
    shear_range=0.01,
    zoom_range=[0.9, 1.1],
    validation_split=0.1,
    #10%를 검증용으로 사용한다. 
    preprocessing_function=preprocess_input
    #mobilenet_v2에 있는 전처리 함수를 사용하겠다.
)

val_datagen = ImageDataGenerator(
    validation_split=0.1,
    preprocessing_function=preprocess_input
)

#데이터를 불러와서 
train_gen = train_datagen.flow_from_directory(
    BASE_PATH,
    target_size=(224, 224),
    #특정 크기로 입력한다. 
    classes=['Full', 'Free'],#label
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    subset='training'
)
#Found 2937 images belonging to 2 classes.

val_gen = val_datagen.flow_from_directory(
    BASE_PATH,
    target_size=(224, 224),
    classes=['Full', 'Free'],
    class_mode='categorical',
    batch_size=32,
    shuffle=False,
    subset='validation'
)
#Found 325 images belonging to 2 classes.

print(val_gen.class_indices)
#{'Full': 0, 'Free': 1}

#Load model for transfer learning(전이학습)
base_model = MobileNetV2(input_shape=(224, 224, 3),
                         weights='imagenet',
                         include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
output = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

model.summary()
