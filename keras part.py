# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 13:48:32 2019

@author: STU24
"""

######################################################################
#11/28#
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import keras


data_datagen = ImageDataGenerator(
        rotation_range = 10, #이미지 회전 범위 (각도) (0 ~ 180)
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        #그림을 수평 또는 수직으로 랜덤하게 평행 이동시키는 범위
        #(원본 가로, 세로 길이에 대한 비율값)
        shear_range = 0.1, #y축 방향으로 각도를 증가시켜 이미지를 변형시킨다.
        zoom_range = 0.1, #확대, 축소 범위: 
        #[lower, upper] = [1-zoom_range, 1+zoom_range]
        horizontal_flip = True, 
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

#rescale을 사용하지 않고 미리 255로 나누어 할 수 도 있다. 
#미리 흑백으로 바꾸고 해도 학습이 잘 된다 

from PIL import Image
import matplotlib.pyplot as plt
import os, glob

#이미지를 불러오자 
img = load_img("C:\\WorkSpace\\101_ObjectCategories\\chair\\image_0001.jpg")
plt.imshow(img)
x = img_to_array(img)#이미지를 np.array로 
x.shape
# (300,233,3) 4차원으로 조정하자 
x = x.reshape((1,) + x.shape)
x.shape
# (1, 300, 223, 3)

#로딩한 이미지들을 ImageDataGenerator의 설정대로 변형하고 저장하자 
i = 0
for batch in data_datagen.flow(x,save_to_dir = 
                               "C:\\WorkSpace\\101_ObjectCategories\\chair",
                               save_prefix = '20191128',
                               save_format = 'jpg'):
    i += 1
    if i > 3:
        break

#이번에는 위의 과정을 반복문으로 실행해보자  
caltech_dir = "C:\\WorkSpace\\101_ObjectCategories\\"
categories = ['chair','camera','butterfly','elephant','flamingo']
nb_class = len(categories)

for idx, cat in enumerate(categories):
    
    image_dir = caltech_dir + cat
    files = glob.glob(image_dir + "\\*.jpg")
    
    #이제 차례대로 이미지 파일을 불러와서 변형한 후에 저장하자 
    for f in files:
        img = load_img(f)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        i = 0
        for batch in data_datagen.flow(x,
                    save_to_dir = image_dir,
                    save_prefix = '20191128', save_format = 'jpg'):
            i += 1
            if i > 3:
                break
            
#이미지를 불려놓았다. 그럼 어제 했던것을 train set과 test set으로 나누어서 해보자
#그리고 scale 도 고려해보자 
                
#이미지 크기는 64 * 64
image_w = 64
image_h = 64
pixels = image_w * image_h * 3

X = []#image
Y = []#label 

for idx, cat in enumerate(categories):
    label = [0 for i in range(nb_class)]
    
    label[idx] = 1
    
    image_dir = caltech_dir + cat
    files = glob.glob(image_dir + "\\*.jpg")
    #print(files)
    
    #이제 차례대로 이미지 파일을 불러와서 저장하자 
    for i , f in enumerate(files):
        #print(i, f)
        img = Image.open(f)
        img = img.convert("RGB")
        img = img.resize((image_w, image_h))
        data = np.asarray(img)
        X.append(data)
        Y.append(label)
        
        if i % 10 == 0:
            print(i, "\n",data)
            
#train set과 test set을 나누어 보자 
from sklearn.model_selection import train_test_split
X = np.array(X)
X.shape
#(1651,64,64,3)
Y = np.array(Y)
Y.shape
#(1651,5)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state = 0)
X_train.shape #(1238, 64, 64, 3)
X_test.shape  #(413, 64, 64, 3)
Y_train.shape #(1238, 5)
Y_test.shape  #(413, 5)

#파일을 저장하기 
image_data = (X_train, X_test, Y_train, Y_test)
np.save("C:\\WorkSpace\\Python_Space\\data\\image_data.npy",
        image_data)

X_train, X_test, Y_train, Y_test = np.load(
        "C:\\WorkSpace\\Python_Space\\data\\image_data.npy",
        allow_pickle = True)

x = tf.placeholder(tf.float32, [None, 64,64,3])
y = tf.placeholder(tf.float32, [None, 5])

#1차 
#2차 drop out 을 추가해서 시도함 
#3차 스케일링, 이미지 데이터들을 255를 나누어서 해 보자 
X_train.max()
X_train.min()
(X_train/255).max()
(X_train/255).min()
Y_train.max()
Y_train.min()
(Y_train/255).max()
(Y_train/255).min()

X_train = X_train/255
X_test = X_test/255


w1 = tf.Variable(tf.random_normal([3,3,3,32], stddev = np.sqrt(2/32)))
print(w1)
#<tf.Variable 'Variable:0' shape=(3, 3, 3, 32) dtype=float32_ref>
L1 = tf.nn.conv2d(x,w1,strides = [1,1,1,1], padding = 'SAME')
print(L1)
#Tensor("Conv2D:0", shape=(1, 64, 64, 32), dtype=float32)
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize = [1,1,1,1], strides = [1,2,2,1],padding='SAME')
print(L1)
#Tensor("MaxPool:0", shape=(?, 32, 32, 32), dtype=float32)

L1 = tf.nn.dropout(L1,0.8)

w2 = tf.Variable(tf.random_normal([3,3,32,64],#32개로 받고 , 64개로 출력해 보자
                                  stddev = np.sqrt(2/32)))

L2 = tf.nn.conv2d(L1, w2, strides = [1,1,1,1], padding = 'SAME')
print(L2)
#Tensor("Conv2D_2:0", shape=(?, 32, 32, 64), dtype=float32)
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize = [1,2,2,1],
                    strides = [1,2,2,1], padding = 'SAME')
print(L2)
#Tensor("MaxPool_1:0", shape=(?, 16, 16, 64), dtype=float32)

L2 = tf.nn.dropout(L2,0.8)

w3 = tf.Variable(tf.random_normal([3,3,64,64],#64개로 받고 64개로 출력해보자 
                                  stddev = np.sqrt(2/64)))
L3 = tf.nn.conv2d(L2, w3, strides = [1,1,1,1], padding = 'SAME')
print(L3)
#Tensor("Conv2D_3:0", shape=(?, 16, 16, 64), dtype=float32)

L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
print(L3)
#Tensor("MaxPool_2:0", shape=(?, 8, 8, 64), dtype=float32)

L3 = tf.nn.dropout(L3,0.8)

w4 = tf.Variable(tf.random_normal([8*8*64,256], stddev = np.sqrt(2/(8*8*64))))

L4 = tf.reshape(L3, [-1, 8*8*64])
print(L4)
#Tensor("Reshape:0", shape=(?, 4096), dtype=float32)
L4 = tf.nn.relu(tf.matmul(L4, w4))
print(L4)
#Tensor("Relu_3:0", shape=(?, 256), dtype=float32)

w5 = tf.Variable(tf.random_normal([256,5],# 5는 분류할 갯수 
                                  stddev = np.sqrt(2/256)))

model = tf.matmul(L4,w5)
print(model)
#Tensor("MatMul_1:0", shape=(?, 5), dtype=float32)
hypothesis = tf.nn.softmax(model)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits = model, labels = y))

optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

batch_size = 100

for epoch in range(1,50):
    avg_cost = 0
    for i in range(int(np.ceil(len(X_train)/batch_size))):
        #표본을 추출한다. 
        x_ = X_train[batch_size * i : batch_size * (i + 1)]
        y_ = Y_train[batch_size * i : batch_size * (i + 1)]
        
        _, cost_val = sess.run([optimizer, cost], feed_dict = {x:x_, y:y_})
    
        avg_cost +=cost_val
        
    print('Epoch:','%04d'%(epoch), 'cost: ','{:.9f}'.format(avg_cost/len(X_train)))
    
is_correct = tf.equal(tf.argmax(hypothesis,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도 : ' , sess.run(accuracy, feed_dict = {x:X_train, y:Y_train}))  
print('정확도 : ' , sess.run(accuracy, feed_dict = {x:X_test, y:Y_test}))  
#test를 넣으니 정확도가 약 78.7%이 나온다.
#drop out 를 하니 약 80.4%가 나온다, 약 74.3%가 나온다.
#He초기값을 사용하니 약 80.6%가 나온다.
#255로 나누어서 하니 정확도가 약72.4% 나온다. 

#스케일링이 이번 데이터에는 의미가 없다.

#############
np.sqrt(2 / 32)
def layer(x,size):
    w = tf.Variable(tf.random_normal(size, stddev = 0.01))
    L = tf.nn.conv2d(x,w,strides = [1,1,1,1], padding = 'SAME')
    L = tf.nn.relu(L)
    L = tf.nn.max_pool(L, ksize = [1,1,1,1],
                       strides = [1,2,2,1], padding = 'SAME')
    return L

def layer2(x,size,n):
    w = tf.Variable(tf.random_normal(size, stddev = 0.01))
    L = tf.nn.conv2d(x,w,strides = [1,1,1,1], padding = 'SAME')
    L = tf.nn.relu(L)
    L = tf.nn.max_pool(L, ksize = [1,1,1,1],
                       strides = [1,2,2,1], padding = 'SAME')
    L = tf.nn.dropout(L,n)
    
    return L

def train(x_data,x_test,y_data,y_test):
    
    x = tf.placeholder(tf.float32, [None, 64,64,3])
    y = tf.placeholder(tf.float32, [None, 5])
    
    L1 = layer(x,[3,3,3,32])
    L2 = layer(L1,[3,3,32,64])
    L3 = layer(L2,[3,3,64,64])
    
    w4 = tf.Variable(tf.random_normal([8*8*64,256], stddev = 0.01))
    L4 = tf.reshape(L3, [-1, 8*8*64])
    L4 = tf.nn.relu(tf.matmul(L4, w4))
    
    w5 = tf.Variable(tf.random_normal([256,5],# 5는 분류할 갯수 
                                  stddev = 0.01))
    model = tf.matmul(L4,w5)
    hypothesis = tf.nn.softmax(model)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits = model, labels = y))
    
    optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    batch_size = 100

    for epoch in range(1,50):
        avg_cost = 0
        for i in range(int(np.ceil(len(x_data)/batch_size))):
            #표본을 추출한다. 
            x_ = x_data[batch_size * i : batch_size * (i + 1)]
            y_ = y_data[batch_size * i : batch_size * (i + 1)]
        
            _, cost_val = sess.run([optimizer, cost], feed_dict = {x:x_, y:y_})
    
            avg_cost +=cost_val
        
        print('Epoch:','%04d'%(epoch), 'cost: ','{:.9f}'.format(avg_cost/len(x_data)))

    is_correct = tf.equal(tf.argmax(hypothesis,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    
    print('정확도 : ' , sess.run(accuracy, feed_dict = {x:x_data, y:y_data}))  
    print('정확도 : ' , sess.run(accuracy, feed_dict = {x:x_test, y:y_test}))
    
train(X_train, X_test, Y_train, Y_test)


#### 선생님의 풀이 ####
import keras
keras.__version__
from keras.models import Sequential
from keras.layers import Conv2D , MaxPooling2D
from keras.layers import Dropout , Flatten ,Dense
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

categories = ["chair","camera","butterfly","elephant","flamingo"]
nb_class = len(categories)

X_train, X_test, Y_train,Y_test = np.load(
        "c:/WorkSpace/Python_Space/data/image_data.npy",
        allow_pickle = True)
X_train =X_train.astype("float")/255
X_test =X_test.astype("float") / 255
X_train.shape 
#(1253, 64, 64, 3)
X_test.shape
#(418, 64, 64, 3)
Y_train.shape
#(1253, 5)
Y_test.shape
#(418, 5)

#model을 형성하자 
#Sequential : layer들을 선으로 쌓기 
model = Sequential()

#add() : layer instance를 layer stack 꼭대기에 추가한다.

#Conv2D() : 합성곱 
#filters : 필터의 갯수, kernel_size : 필터의 window 크기 (높이, 너비),
#strides : 스트라이드 간격 (높이, 너비), padding : 'valid'나 'same'
#input shape : (batch, channels, rows, columns)
#output shape : (batch, filters, new_rows, new_columns)
model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1, 1),
                 padding = "same",
                 input_shape =(64,64,3), activation='relu'))

#MaxPooling2D() : 최대 풀링 
#pool_size : 값을 뽑아내는 범위 (수직, 수평)
model.add(MaxPooling2D(pool_size=(2,2)))

#model.add(Dropout(0.25))#25%를 정지함 


#summary() : network의 요약 출력
#
model.summary()#다음층으로 갈때 어떻게 가는지 나옴 
#_________________________________________________________________
#Layer (type)                 Output Shape              Param #   
#=================================================================
#conv2d_1 (Conv2D)            (None, 64, 64, 32)        896       
#_________________________________________________________________
#max_pooling2d_2 (MaxPooling2 (None, 32, 32, 32)        0         
#=================================================================
#Total params: 896
#Trainable params: 896
#Non-trainable params: 0
#_________________________________________________________________

#param이 무엇을 의미하는 건지 모르겠다. 


model.add(Conv2D(filters=64,kernel_size=(3,3),padding ="same",
                 activation='relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
#model.add(Dropout(0.25))

model.summary()
#_________________________________________________________________
#Layer (type)                 Output Shape              Param #   
#=================================================================
#conv2d_1 (Conv2D)            (None, 64, 64, 32)        896       
#_________________________________________________________________
#max_pooling2d_2 (MaxPooling2 (None, 32, 32, 32)        0         
#_________________________________________________________________
#conv2d_2 (Conv2D)            (None, 32, 32, 64)        18496     
#_________________________________________________________________
#max_pooling2d_3 (MaxPooling2 (None, 16, 16, 64)        0         
#=================================================================
#Total params: 19,392
#Trainable params: 19,392
#Non-trainable params: 0
#_________________________________________________________________

model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
#model.add(Dropout(0.25))

model.summary()
#_________________________________________________________________
#Layer (type)                 Output Shape              Param #   
#=================================================================
#conv2d_1 (Conv2D)            (None, 64, 64, 32)        896       
#_________________________________________________________________
#max_pooling2d_2 (MaxPooling2 (None, 32, 32, 32)        0         
#_________________________________________________________________
#conv2d_2 (Conv2D)            (None, 32, 32, 64)        18496     
#_________________________________________________________________
#max_pooling2d_3 (MaxPooling2 (None, 16, 16, 64)        0         
#_________________________________________________________________
#conv2d_3 (Conv2D)            (None, 14, 14, 64)        36928     
#_________________________________________________________________
#max_pooling2d_4 (MaxPooling2 (None, 7, 7, 64)          0         
#=================================================================
#Total params: 56,320
#Trainable params: 56,320
#Non-trainable params: 0
#_________________________________________________________________

#Flatten() : input을 납작하게 하는 layer , (None, 64, 32, 32) -> (None, 65536)
model.add(Flatten()) 

#Dense() : output = activation(dot(input, kernel(filter)) + bias) 를 작용
# 
model.add(Dense(512,#512개로 펼치기 
                activation='relu')) 

#Dropout() : input unit를 0으로 하는 비율값을 정함  
model.add(Dropout(0.5)) 

model.summary()

model.add(Dense(nb_class,#512개에서 5개로 
                activation='softmax')) 
model.summary()
#_________________________________________________________________
#Layer (type)                 Output Shape              Param #   
#=================================================================
#conv2d_1 (Conv2D)            (None, 64, 64, 32)        896       
#_________________________________________________________________
#max_pooling2d_2 (MaxPooling2 (None, 32, 32, 32)        0         
#_________________________________________________________________
#conv2d_2 (Conv2D)            (None, 32, 32, 64)        18496     
#_________________________________________________________________
#max_pooling2d_3 (MaxPooling2 (None, 16, 16, 64)        0         
#_________________________________________________________________
#conv2d_3 (Conv2D)            (None, 14, 14, 64)        36928     
#_________________________________________________________________
#max_pooling2d_4 (MaxPooling2 (None, 7, 7, 64)          0         
#_________________________________________________________________
#flatten_1 (Flatten)          (None, 3136)              0         
#_________________________________________________________________
#dense_1 (Dense)              (None, 512)               1606144   
#_________________________________________________________________
#dropout_1 (Dropout)          (None, 512)               0         
#_________________________________________________________________
#dense_2 (Dense)              (None, 5)                 2565      
#=================================================================
#Total params: 1,665,029
#Trainable params: 1,665,029
#Non-trainable params: 0
#_________________________________________________________________

# 'binary_crossentropy' : 이진 분류,
# 'categorical_crossentropy' : 다중분류,  회귀 : mse, mae

#compile() : train을 위한 model의 환경설정 
model.compile(loss='categorical_crossentropy',#cross entropy
              optimizer= 'adam',#adam 
              metrics = ['accuracy'])

#fit() : 주여진 epoch의 갯수만큼 model을 train함 
model.fit(X_train,Y_train,batch_size = 100,epochs = 30)

score =model.evaluate(X_test,Y_test)
print('loss : ',score[0])
print('accuracy : ' ,score[1])
#처음에 scale 안 하고했더니 정확도가 18%나왔다.
#scale하고 했더니 정확도가 80.8% 나왔다.

plt.imshow(X_test[100],cmap='Greys')
data = X_test[100].reshape([1,64,64,3])
print("Prediction : ",categories[np.argmax(model.predict(data))])


img = Image.open("c:/WorkSpace/elephant test 1.jpg")
plt.imshow(img)
plt.imshow(img.resize([64,64]))
img = img.convert("RGB")
data = img.resize([64,64])
data = np.asarray(data)
data.shape
data = data.reshape([1,64,64,3])
data.shape
data = data/255
print("Prediction : ",categories[np.argmax(model.predict(data))])
