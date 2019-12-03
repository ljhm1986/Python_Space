# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 18:28:36 2019

@author: STU24
"""
import numpy as np
import sys, os
#부모 디렉터리의 파일을 가져올 수 있도록 설정
sys.path.append(os.pardir)
#from dataset.mnist import load_mnist

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#다음을 실행하면 불러지면서 동시에 해당 경로에 파일이 저장된다. 
mnist = input_data.read_data_sets(
        'C:\\WorkSpace\\Python_Space\\data\\MNIST_data\\',one_hot = True)

from PIL import Image

def img_show(img):
    #numpy array를 image 객체로 변환한다 
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()
    

np.shape(mnist.train.images)
np.shape(mnist.train.labels)
np.shape(mnist.test.images)
np.shape(mnist.test.labels)

img = mnist.train.images[0]
label = mnist.train.labels[0]
print(label)

print(img.shape)
#원래 형태로 바꾸어주자 
img = img.reshape(28, 28)
print(img.shape)

img_show(img)

def get_data():
    ()

#
import pickle
#미리 학습된 weight, bias값들이 저장된 파일에서 불러오자   
def init_network():
    with open("C:\\WorkSpace\\Python_Space\\data\\sample_weight.pkl",'rb') as f:
        network = pickle.load(f)
    
    return network

def sigmoid(x):
    return 1/(1+np.exp(-x))

def predict(network,x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x ,W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1 ,W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2 ,W3) + b3
    y = sigmoid(a3)
    
    return y


network = init_network()
accuracy_cnt = 0
for i in range(len(mnist.test.images)):
    y = predict(network,mnist.test.images[i])
    p = np.argmax(y)#확률이 가장 높은 인덱스를 얻는다
    s = np.argmax(mnist.test.labels[i])#정답인 부분의 인덱스를 얻는다.
    if p == s:#일치하면 맞춘 갯수에 추가한다. 
        accuracy_cnt += 1

print('Accuracy: '+ str(float(accuracy_cnt) / len(mnist.test.images)))
#93.52%가 나온다.     
   
#batch 처리를 해서 실행해보자 
network = init_network()

batch_size = 100 #배치 크기
accuracy_cnt = 0

for i in range(0, len(mnist.test.images),batch_size):
    x_batch = mnist.test.images[i:i+batch_size]
    y_batch = predict(network, x_batch)
    
    #0,1 번째 차원중에서 1번째 차원의 최대값 인덱스를 찾자 
    p = np.argmax(y_batch, axis = 1)
    s = np.argmax(mnist.test.labels[i:i+batch_size], axis = 1)
    accuracy_cnt += np.sum(p == s)
    
print('Accuracy: '+ str(float(accuracy_cnt) / len(mnist.test.images)))
#93.52%가 나온다.

