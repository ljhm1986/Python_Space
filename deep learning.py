# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 11:48:21 2019

@author: stu12
"""
#딥러닝(Deep Neural Network)
##11/18##

#AND 
#X1  	X2  	Y
#----	----	----
#0	    0	    0
#1	    0	    0
#0	    1	    0
#1    	1	    1

#w1 = 0.5, w2 = 0.5 , theta = =0.7
w1 = 0.5
w2 = 0.5
theta = 0.7
def AND(x1,x2):
    result = w1 * x1 + w2 * x2
    if result > theta:
        return 1
    else:
        return 0
    
AND(0,0)
AND(1,0)
AND(0,1)
AND(1,1)
# 실제 W1, W2 theta 값을 찾아서 해야 한다.

import numpy as np
inputData = np.array([[0,0],[0,1],[1,0],[1,1]])
inputData.shape
inputData[0][1]

print('AND Perceptron')
"""
[0,0] => 0
[0,1] => 0
[1,0] => 0
[1,1] => 1 """

def AND2(array):
    for i in array:
        temp = AND(i[0], i[1])
        print("[{}, {}] => {}".format(i[0],i[1],temp))
            
AND2(inputData)

for i in inputData:
    print(str(i) + " => " + str(AND(i[0],i[1])))
    
#np의 sum 을 이용하려면?
w = np.array([w1, w2])

def AND3(x1, x2):
    x = np.array([x1, x2])
    temp = np.sum(w*x)
    if  temp > theta:
        return 1
    else:
        return 0
   
AND3(1,1)     
"""
OR
X1	    X2      Y
----	----	----
0	    0	    0
1	    0	    1
0	    1	    1
1	    1	    1 """

def OR(x1,x2):
    x = np.array([x1, x2])
    temp = np.sum(w * x)
    
    if temp > 0:
        return 1
    else:
        return 0
    
for i in inputData:
    print(str(i) + " => " + str(OR(i[0],i[1])))
    
#NAND(NOT AND)
#X1	    X2	    Y
#----	----	----
#0	    0	    1
#1	    0	    1
#0	    1	    1
#1	    1	    0

def NAND(x1, x2):
    x = np.array([x1, x2])
        
    temp = np.sum(w * x)
        
    if temp > theta:
        return 0
    else: 
        return 1

for i in inputData:
    print(str(i) + " => " + str(NAND(i[0],i[1])))
    
#XOR(eXclusive OR)
#X1	    X2	    Y
#----	----	----
#0	    0	    0
#1	    0	    1
#0    	1	    1
#1	    1	    0
#XOR(NOT OR)는 단일 회로로는 안 된다.
#다층퍼셉트론(Multi layer Perceptron)
#X1	    X2	    OR	    NAND	AND(OR와 NAND의)
#----	----	----	----	----
#0	    0	    0	    1	    0
#1	    0	    1	    1	    1
#0	    1	    1	    1	    1
#1	    1	    1	    0	    0

def XOR(x1, x2): #0층
    s1 = OR(x1, x2) #1층
    s2 = NAND(x1, x2)#1층
    result = AND(s1,s2)#2층
    
    return result

for i in inputData:
    print(str(i) + " => " + str(XOR(i[0],i[1])))
    
#지금까지는 
w1 = 0.5
w2 = 0.5
b = -0.5

def AND4(x1, x2):
    w = np.array([w1, w2, b])
    x = np.array([x1, x2, 1])
    temp = np.sum(w*x)
    
    if temp > 0:
        return 1
    else:
        return 0      
    
for i in inputData:
    print(str(i) + " => " + str(AND4(i[0],i[1])))

b = -0.2
def OR2(x1, x2):
    w = np.array([w1, w2, b])
    x = np.array([x1, x2, 1])
    temp = np.sum(w*x)
    
    if temp > 0:
        return 1
    else:
        return 0 
    
for i in inputData:
    print(str(i) + " => " + str(OR2(i[0],i[1])))
    
def NAND2(x1, x2):
    w1, w2, b = -0.5, -0.5, 0.7
    w = np.array([w1, w2, b])
    x = np.array([x1, x2, 1])
        
    temp = np.sum(w * x)
        
    if temp > 0:
        return 1
    else: 
        return 0

for i in inputData:
    print(str(i) + " => " + str(NAND2(i[0],i[1])))
    

#step ft의 함수를 만들자 
def step_function(x):
    
    if x > 0:
        return 1
    else:
        return 0
    
step_function(4)
step_function(10)
step_function(-2)

def step_function2(array):
    
    for i in array:
        if i > 0:
            print(1)
        else:
            print(0)
            
step_function2(np.array([1]))
step_function2(np.array([1,-1,10]))

[1 if i > 0 else 0 for i in np.array([1,-1,10])]

x = np.array([1,10,-1])
y = x > 0
y
y.astype(np.int)

#bool -> int변환 : True -> 1, False -> 0

def step_ft3(array):
    y = x > 0
    return y.astype(np.int)

step_ft3(x)

#step ft의 그래프를 그려보자
import matplotlib.pylab as plt
x = np.arange(-5.0,5.0,0.1)
x
y = step_ft3(x)
y
plt.plot(x,y)

#sigmoid ft의 그래프를 그려보자
x = np.arange(-10.0,10.0,0.1)
x
y = 1 / (1 + np.exp(-x))
y
plt.plot(x,y)

def sigmoid(x):
    
    return 1 / (1 + np.exp(-x))

sigmoid(x)


#numpy 복습한 부분은 numpy.py로 이동 

#######################################################################
#11/20#
import numpy as np
import matplotlib.pylab as plt

#y = wx + b 를 구해보기
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.array([1,2])
w = np.array([[3,2],[1,4]])
b = np.array([1,1])

y = np.dot(w,x) + b
y
z = sigmoid(y)
z

#plt.plot(x,y)

## 다음은 층이 2개이다. 
x = np.array([1,2])
w1 = np.array([[4,5,6],[7,8,9]])
w2 = np.array([[2,3],[4,5],[3,2]])
b1 = np.array([2,1,3])
b2 = np.array([2,2])

x.shape
x.ndim

x[0]
w1[0][0]
print(x)
print(w1)
print(b1)

#은닉층
y1 = np.dot(x, w1) +b1
#y1 = [1 2] . [[4 5 6]  + [2 1 3]
#              [7 8 9]]
print(y1)
z1 = sigmoid(y1)
z1
#출력층
y2 = np.dot(z1, w2) + b2
y2
z2 = sigmoid(y2)
z2

###### 단순한 형태
#입력값
x = np.array([1,2])
#weight
w = np.array([[1,3,5],[2,4,6]])
#출력값
y1 = np.dot(x,w)
y1

#또는 
x = np.array([[1],[2]])
w = np.array([[1,2],[3,4],[5,6]])
y2 = np.dot(w,x)
y2


####

def init_network():
    network = {}
    network['w1'] = np.array([[0.1, 0.3, .5],[0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['w2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['w3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    
    return network

def forword(x,W,b):
    a = np.dot(x,W) + b
    z = sigmoid(a)
    
    return z

def identity_function(x):
    return x

def forword2(x,W,b):
    a = np.dot(x,W) + b
    y = identity_function(a)
    
    return y

network = init_network()
x = np.array([1.0, 0.5])
y1 = forword(x,network['w1'],network['b1'])
y2 = forword(y1,network['w2'],network['b2'])
y3 = forword2(y2,network['w3'],network['b3'])
print(y3)



#[문제] 입력값이 0이 넘으면 그 입력값으로 return 해주고 
#0 이하면 0을 return 해주는 relu함수를 작성해주세요

def relu(x):
    
    if x > 0:
        return x
    else:
        return 0
   
relu(6)
relu(-3)
x = np.array([-1,0,1,2,3,4,5])
relu(x)#안됨

x.ndim
x.shape
x.dtype

y = np.array([[-5,9,3],[0,-2,3]])
y.shape
y.size

def relu2(x):
    
    x = np.maximum(0,x)
    
    return x
            
relu2(x)       
relu2(y)

#conda install tensorflow
import tensorflow as tf

#버전 확인 
tf.__version__

#상수 선언(객체 생성)
hello = tf.constant("hello")
hello

#세션을 열어야 한다. 
#sess = tf.Session()
session = tf.compat.v1.Session()
#버전이 2.0.0 이라서 명령어가 다르다.

#conda install tensorflow==1.15
#버전 1.15.0 을 재설치했다.
sess = tf.Session()
sess.run(hello)

a = tf.constant(1234)
b = tf.constant(5678)

add_op = a + b
sub_op = a - b
res = sess.run(add_op)
res1 = sess.run(sub_op)
res
print(res)
print(res1)

#상수선언 
x1 = tf.constant(120, name = 'x1')
x2 = tf.constant(130, name = 'x2')
x3 = tf.constant(140, name = 'x3')

#변수선언
v = tf.Variable(0, name = 'v')

calc_op = x1 + x2 + x3

#값을 지정해 놓기 
assign_op = tf.assign(v, calc_op)

sess = tf.Session()
sess.run(assign_op)#390
sess.run(v)#390 이 나온다.

#다르게 해 보면
v1 = tf.Variable(0, name = 'v1')
v1 = calc_op
sess.run(v1)#390

#변수를 run할때 정하고 싶을때
p1 = tf.placeholder('int32')
p2 = tf.placeholder('int32')
print(p1)

y = tf.add(p1, p2)
sess = tf.Session()
#run할때 값을 넣자
sess.run(y, feed_dict = {p1: 10, p2:3})

#함수        설명
#tf.add      덧셈
#tf.subtract 뺄셈
#tf.multiply 곱셈
#tf.div      나눗셈의 몫, 소수점
#tf.truediw  나눗셈의 몫, 소수점
#tf.mod      나눗셈의 나머지
#tf.abs      절대값
#tf.negative 음수를 리턴
#tf.sign     부호를 리턴 (음수는 -1, 양수는 1, 0은 0리턴)
#tf.reciprocal 역수를 리턴
#tf.square   제곱을 계산
#tf.round    반올림
#tf.sqrt     제곱근 계산
#tf.pow      거듭제곱
#tf.exp      지수값
#tf.log      로그값
#tf.maximum  최대값
#tf.minimun  최소값

a = tf.placeholder(tf.int32, [3])
a
b = tf.constant(2)
x2_OP = a + b
sess = tf.Session()
r1 = sess.ru(v2)


#
x = tf.placeholder('float32')
y = tf.placeholder('float32')
z = tf.multiply(x,y)

sess = tf.Session()
print(sess.run(z, feed_dict = {x:[3,4], y:[5,6]}))
print(sess.run(z, feed_dict = {x:[2,3,4], y:[5,6,7]}))

#
x1 = tf.constant([[1,2,3],[4,5,6]])
y1 = tf.constant([[1,2],[3,4],[5,6]])
z1 = tf.Variable(0)
z1 = tf.matmul(x1, y1)

sess = tf.Session()
sess.run(z1)
sess.close()

##
x = tf.placeholder(tf.float32, shape = (2,3))
y = tf.placeholder(tf.float32, shape = (3,2))
z = tf.matmul(x,y)

sess = tf.Session()
#아래 실행하면 kernel 관련 에러가 뜬다....??
#An error ocurred while starting the kernel
#
#2019󈚯󈚸 16:15:46.310397: I tensorflow/core/platform/cpu_feature_guard.cc:145] 
#This TensorFlow binary is optimized with Intel(R) MKL‑DNN to use the following 
#CPU instructions in performance critical operations: AVX AVX2
#To enable them in non‑MKL‑DNN operations, rebuild TensorFlow with the 
#appropriate compiler flags.
#2019󈚯󈚸 16:15:46.310397: I tensorflow/core/common_runtime/process_util.cc:115] 
#Creating new thread pool with default inter op setting: 8. Tune using 
#inter_op_parallelism_threads for best performance.
#
#tensorflow를 설치했던 cmd 창을 닫으니까 에러가 나지 않는다 ...
sess.run(z, feed_dict = {x:[[1,2,3],[4,5,6]], y: [[1,2],[3,4],[5,6]]})

#[문제] 상수를 이용해서 아래와 같이 결과를 출력하는 프로그램을 작성하세요
#a + b = 6
#a * b = 8

a = tf.constant(2)
b = tf.constant(4)

z1 = tf.add(a,b)
z2 = tf.multiply(a,b)

sess = tf.Session()
sess.run(z1)
sess.run(z2)
sess.close()

c = a + b
d = a * b
#open - close를 한 번에 수행하기 
with tf.Session() as sess:
    print("a + b = {}".format(sess.run(c)))
    print("a * b = {}".format(sess.run(d)))
    
#변수로 선언한 경우 
a = tf.Variable(2)
b = tf.Variable(4)
c = a + b
d = a * b
with tf.Session() as sess:
    print("a + b = {}".format(sess.run(c)))
    print("a * b = {}".format(sess.run(d)))
#작동하지 않는다.
    
#with문에 전역 변수 초기화를 추가해야 함 
a = tf.Variable(2)
b = tf.Variable(4)
c = a + b
d = a * b
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("a + b = {}".format(sess.run(c)))
    print("a * b = {}".format(sess.run(d)))

    
#[문제] 입력값을 실행시에 입력하도록 하는 프로그램을 작성하세요.
#Add : 6
#Multiply : 8

x = tf.placeholder('int32')
y = tf.placeholder('int32')

z1 = tf.add(x,y)
z2 = tf.multiply(x,y)

with tf.Session() as sess:
    print(sess.run(z1, feed_dict = {x:2,y:4}))
    print(sess.run(z2, feed_dict = {x:2,y:5}))


#다음 자료형 shape는 다르다.
x1 = np.array([1,2,3])#3행    
x2 = np.array([[1,2,3]])#3행 1열
