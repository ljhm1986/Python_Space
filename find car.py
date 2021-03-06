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

#데이터를 다운로드한 경로를 입력하고 
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
plt.show()

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
    #mobilenet_v2에 넣을때 사용할 전처리 함수를 사용하겠다.
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

print(train_gen.class_indices)

print(val_gen.class_indices)
#{'Full': 0, 'Free': 1}

#Load model for transfer learning(전이학습)
#MobileNetV2 : keras에서 사용가능한 이미지 분류 모델, 
#ImageNet 데이터셋에서 사전 훈련된 네트워크 중 하나
base_model = MobileNetV2(input_shape=(224, 224, 3),
                         #모델을 초기화할 가중치 체크포인트를 지정
                         weights='imagenet',
                         #네트워크의 최상위 완전 연결 분류기의 포함 여부  
                         include_top=False)
#output은 직접 정의할 것이라서

#
x = base_model.output
#1차원으로 데이터를 펴줌
x = GlobalAveragePooling2D()(x)
#output은 2종류 
output = Dense(2, activation='softmax')(x)

#Model(Network) : network에 훈련과 평가과정을 추가함 
#입력과 출력 텐서만으로 Model 객체를 생성 
#이 경우에는 MobileNetV2 네트워크로 입력하고 output으로 출력함 
model = Model(inputs=base_model.input, outputs=output)

#환경설정
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc'])

model.summary()

#model의 층들이 학습이 가능하도록 설정하자 
for layer in model.layers:
    layer.trainable = True#False이면 학습이 진행 안됨

model

#학습을 시켜보자, history객체에 훈련하는 동안 발생한 정보들을 저장한다. 
history = model.fit_generator(
    train_gen,
    validation_data=val_gen,#검증 데이터 전달, epoch마다 손실과 정확도를 계산함 
    epochs=10,
    #steps_per_epoch = ceil(),
    callbacks=[
        #ModelCheckpoint : 모든 epoch 이후에 model의 가중치를 저장한다. 
        ModelCheckpoint('model.h5',
                        #아래 둘은 
                        monitor='val_acc',
                        save_best_only=True,
                        verbose = 1)
    ]
)
#시간이 많이 걸리네 
#집에서는 작동했는데 학원 컴퓨터로 작동하지 않는다. 이유가 뭐지 ?
#학원 karas version = 2.2.4
#집 karas version = 2.3.1
#steps_per_epoch : 정수.
#한 epoch 종료를 선언하고 다음 epoch를 시작하기 전에, generator에서 산출될 총 단계의 수
#

#모델 저장한걸 불러들이자 
model = load_model('model.h5')

#마지막 layer의 weights를 불러오기
last_weight = model.layers[-1].get_weights()[0] 
last_weight.shape  # (1280, 2)
# __________________________________________________________________________________________________
# global_average_pooling2d_1 (Glo (None, 1280)         0           out_relu[0][0]

# __________________________________________________________________________________________________
# dense_1 (Dense)                 (None, 2)            2562        global_average_pooling2d_1[0][0]
# ==================================================================================================
#을 보면 free connected layer이니까 1280에 2를 곱해주는 것이다. 행렬을 곱해주는 거라 생각하면 

#입력에 대하여 원본 model의 활성화값을 반환하는 model 
new_model = Model(
    inputs=model.input,
    outputs=(
        #마지막에서 3번째 layer의 출력을 추출함, out_relu (ReLU) layer
        model.layers[-3].output, 
        # the layer just before GAP, for using spatial features
        #convolution layer까지가 spatial feature를 가지기 때문이다.
        #이것은 global_average_pooling을 지나면 사라진다.
        #마지막 layer의 출력을 추출함, dense_1 (Dense) layer 
        model.layers[-1].output 
    )
)

new_model.summary()

'''
Free/img_129173058.jpg
Free/img_723080007.jpg
Free/img_815061601.jpg
Full/img_127040601.jpg
Full/img_809172559.jpg
'''
#PIL객체로 불러들인 후, numpy array로 변환합니다.
test_img = img_to_array(load_img(os.path.join(BASE_PATH,
                                              'Free/img_129173058.jpg'),
                                 target_size=(224, 224)))
test_img.shape # (224, 224, 3)

#차원을 추가하고, 데이터를 전처리합니다.
test_input = preprocess_input(np.expand_dims(test_img.copy(), axis=0))
test_input.shape # (1,224,224,3)

pred = model.predict(test_input)
print(pred)
#[[0.9924616  0.00753842]] 형태로 나옴  

#
plt.figure(figsize=(8, 8))
plt.title('%.2f%% Free' % (pred[0][1] * 100))
plt.imshow(test_img.astype(np.uint8))
plt.show()
#비어있다고 나옴, 어디가 비어있는지는 나오지 않음

#Draw Activation Map
last_conv_output, pred = new_model.predict(test_input)
last_conv_output.shape # (1, 7, 7, 1280)

#array의 형태에서 단일 차원 성분은 제거한다 
last_conv_output = np.squeeze(last_conv_output) # (1, 7, 7, 1280) -> (7, 7, 1280)
last_conv_output.shape 

#마지막 convolution에서 나온걸 이미지 확대 
feature_activation_maps = scipy.ndimage.zoom(
    last_conv_output, (32, 32, 1), order=1) # (7, 7, 1280) -> (224, 224, 1280)
feature_activation_maps.shape # 

#예측값을 알아내기
pred_class = np.argmax(pred) # 0: Full, 1: Free

#global_average_pooling을 건너뛰기 위해서, 이런식으로 추출하고
predicted_class_weights = last_weight[:, pred_class] # (1280, 1)
#바로 곱해준다.
final_output = np.dot(feature_activation_maps.reshape((224*224, 1280)),
 predicted_class_weights).reshape((224, 224)) 
# (224*224, 1280) dot_product (1280, 1) = (224*224, 1)
# 

#class activation map 출력 
plt.imshow(final_output, cmap='jet')
plt.show()

#같이 출력, 빨간색을 집중적으로 보고 비어있는지 차있는지 판단한다.   
fig, ax = plt.subplots(nrows=1, ncols=2)
fig.set_size_inches(16, 20)

ax[0].imshow(test_img.astype(np.uint8))
ax[0].set_title('image')
ax[0].axis('off')

ax[1].imshow(test_img.astype(np.uint8), alpha=0.5)
ax[1].imshow(final_output, cmap='jet', alpha=0.5)
ax[1].set_title('class activation map')
ax[1].axis('off')
plt.show()