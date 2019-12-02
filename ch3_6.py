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