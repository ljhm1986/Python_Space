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
#import 를 할때 cudart64_100.dll 가 없다는 내용이 나오는데 GPU 사용 설정을 하지 않아서이다. 
#(학원 컴퓨터로 할 때는 아무런 메시지가 안 나왔었는데??)

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
#Session 여는 순간 여러 메시지가 나온다, GPU 관련된 내용이다. 
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

#변수의 값을 지정해 놓기 
assign_op = tf.assign(v, calc_op)

sess = tf.Session()
sess.run(assign_op)#390
sess.run(v)#390 이 나온다.

#다르게 해 보면
v1 = tf.Variable(0, name = 'v1')
v1 = calc_op
sess.run(v1)#390

#변수를 run할때 정하고 싶을때 인스턴스화 하여 가동
p1 = tf.placeholder('int32')
p2 = tf.placeholder('int32')
print(p1)

#함수를 정함 
y = tf.add(p1, p2)
sess = tf.Session()
#run할때 함수와 값을 넣자, 
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

#3개의 행을 가진 형태로 선언
a = tf.placeholder(tf.int32, [3])
a
b = tf.constant(2)
x2_OP = a + b
sess = tf.Session()



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
# array([[22., 28.],
#        [49., 64.]], dtype=float32

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
    #변수 초기화
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

##################################################################
#11/21#
#텐서 자료 구조
#- 텐서 텐서플로의 기본 자료 구조
#- 텐서 다차원 배열, 리스트로 구성
#
#1차원 텐서
import numpy as np
import tensorflow as tf

arr = np.array([1,2,3])
arr[0]
arr.ndim
arr.shape #(3,)
arr.dtype

#numpy array 를 ..
with tf.Session() as sess:
    print(sess.run(arr))
#error가 난다, 변환작업을 해 줘야 한다.
arr_tf = tf.convert_to_tensor(arr, dtype = tf.int32)
with tf.Session() as sess:
    sess.run(arr_tf)
    print(sess.run(arr_tf[0]))

arr_tf.dtype
#tf.int32
arr_tf.shape
#TensorShape([Dimension(3)])
arr_tf.ndim #안 됨

x = np.array([1,2,3]) #(3,)
w = np.array([[2],[2],[2]]) #(3,1)
np.dot(x,w)
print(np.dot(x,w))

#tensor로 해 보자 
x1 = tf.constant([1,2,3])
w1 = tf.constant([[2],[2],[2]])
y = tf.matmul(x1,w1) #안 됨, x1을 명확하게 (1,3)되도록 수정해야 한다. 

x1 = tf.constant([[1,2,3]])
w1 = tf.constant([[2],[2],[2]])
y = tf.matmul(x1,w1)

x1.shape
print(x1.shape)
x1.get_shape
print(x1.get_shape())
sess = tf.Session()
sess.run(y)
sess.close()

#변수일 경우
x1 = tf.Variable([[1,2,3]])
w1 = tf.Variable([[2],[2],[2]])
y = tf.matmul(x1,w1)

x1.shape
print(x1.shape)
x1.get_shape
print(x1.get_shape())
sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(y)

xx = sess.run(x1)
xx #array([[1, 2, 3]]) numpy array처럼 나온다.

sess.close()

#2차원 텐서 
#numpy array 를  tensor에서 사용하기 
arr_1 = np.array([[1,2,3],[4,5,6],[7,8,9]])
arr_2 = np.array([[1,1,1],[2,2,2],[3,3,3]])
type(arr_1)

tm1 = tf.constant(arr_1)
tm2 = tf.constant(arr_2)

tm_product = tf.matmul(tm1, tm2)
tm_add = tf.add(tm1, tm2)

sess = tf.Session()
sess.run(tm_product)
sess.run(tm_add)
sess.close()

#3차원
arr_3 = np.array([[[1,2],[3,4],[2,3]],[[3,2],[1,4],[9,5]]])
arr_3.ndim
arr_3.shape #(plane, row, column)
print(arr_3)

tm_3 = tf.constant(arr_3)
sess = tf.Session()
sess.run(tm_3)

print(tm_3.get_shape())

#[문제] 
#x 변수는 1행 3열 모양의 값은 1,2,3
#w 변수는 3행 1열 모양의 값은 2,2,2
#y 변수는 x와 w를 행렬의 곱을 이용한 결과를 수행하세요
x = np.array([[1,2,3]])
w = np.array([[2],[2],[2]])
y = np.dot(x,w)
y

tf_x = tf.constant(x)
tf_w = tf.constant(w)
tm_product = tf.matmul(tf_x, tf_w)

sess = tf.Session()
sess.run(tm_product)
sess.close()

#
points = [[1,2,3],[4,5,6]]
vectors = tf.constant(points)
print(points)
print(vectors)

#2차원 백터를 3차원으로 확장하자 
expanded_vectors = tf.expand_dims(vectors,0)
print(expanded_vectors)
expanded_vectors.get_shape()
print(expanded_vectors.get_shape())

#0으로 채워넣은 행렬 만들기 
zeros = tf.zeros_like([[0,0,0],[1,2,3]], dtype = tf.int32, name = 'zeros')
zeros

sess = tf.Session()
sess.run(zeros)
#array([[0, 0, 0],
#       [0, 0, 0]])

zeros1 = tf.zeros([2,3], dtype = tf.int32, name = 'zeros1')
sess.run(zeros1)
#array([[0, 0, 0],
#       [0, 0, 0]])

zeros2 = tf.zeros([3,4,5], dtype = tf.int32, name = 'zeros2')
sess.run(zeros2)

#1로 채워넣기 
ones1 = tf.ones([3,4], dtype = tf.int32, name = 'ones1')
sess.run(ones1)

ones2 = tf.ones([3,4,5], dtype = tf.float32, name = 'ones2')
sess.run(ones2)

#특정 숫자로 채워넣기
fill = tf.fill([2,3],7)
sess.run(fill)

x = tf.constant(7, shape = [2,3])
sess.run(x)

#정규분포 난수 추출
norm = tf.random_normal([3,3])
sess.run(norm)

#표준정규분포 난수 추출
z = tf.random_normal([3,3], mean = 0, stddev = 1)
sess.run(z)

#주어진 값들을 shuffle (위치를 임의로 바꾸기)
s = tf.constant([[1,2],[3,4],[5,6]])
shuff = tf.random_shuffle(s)
sess.run(shuff)

#uniform 분포에서 추출
unif = tf.random_uniform([2,3], minval = 1, maxval = 3)
sess.run(unif)

#seed를 고정해서 추출하기 
a = tf.random_uniform([1], seed = 1)
b = tf.random_normal([1])
print("session1")
with tf.Session() as sess1:
    print(sess1.run(a))
    print(sess1.run(a))
    print(sess1.run(b))
    print(sess1.run(b))

print("session2")
with tf.Session() as sess1:
    print(sess1.run(a))
    print(sess1.run(a))
    print(sess1.run(b))
    print(sess1.run(b))  

#다른 session에서 a의 값들이 같게 나온다. b는 다르게 나온다. 

#처음부터 seed를 고정해서 추출하기 
tf.set_random_seed(1234)
x = tf.random_uniform([1])
y = tf.random_normal([1])
print("session1")
with tf.Session() as sess1:
    print(sess1.run(x))
    print(sess1.run(x))
    print(sess1.run(y))
    print(sess1.run(y))
    
print("session2")
with tf.Session() as sess1:
    print(sess1.run(x))
    print(sess1.run(x))
    print(sess1.run(y))
    print(sess1.run(y))  

#수열 값을 표현, 단 실수형으로만 표시    
lin = tf.linspace(10., 12., 5,#숫자들의 총 갯수
                  name = "linspace")
sess.run(lin)

#정수로 수열을 만드려면 
ran = tf.range(start = 1, limit = 10,
               delta = 1)#숫자의 간격
sess.run(ran)

ran = tf.range(11)
sess.run(ran)

#y = in * w + b
inputData = tf.Variable(tf.fill([1,32],7.))
weight = tf.Variable(tf.random_normal(
        [32,32],mean = 0, stddev = 1, name = 'weight'))

bias = tf.Variable(tf.zeros([32]), name = 'bias')

weight.get_shape()
bias.get_shape()

y = tf.matmul(inputData, weight) + bias
y.shape


sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(weight)
sess.run(bias)
sess.run(y)
sess.close()
sess1.close()
dir()

#변수 초기화 
w1 = tf.Variable(tf.random_normal([3,3], mean = 0, stddev = 1, name = 'w1'))
w2 = tf.Variable(w1.initialized_value(), name = 'w2')
w_twice = tf.Variable(w1.initialized_value() * 2.0, name = "w_twice")

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(w1)
sess.run(w2)
sess.run(w_twice)
sess.close()

#------------
#시간	점수
#----	----
#2	71
#4	83
#6	91
#8	97
#
#3시간 공부하면 몇 점을? 
#y = a * x + b
#y : 종속변수, 목표변수
#x : 독립변수, 설명변수
#a : 기울기
#b : 절편
#최소제곱법(method of least squares)
#- 일차함수의 기울기 a, 절편 b
#- 회귀분석에서 사용하는 표준방식
#- 실험이나 관찰을 통해 얻은 데이터를 분석하여 
#미지의 상수를 구할 때 사용된다.
#
#	        Σ(x - x평균)(y - y평균)
#기울기 = ---------------------------
#	             Σ(x - x평균)^2
#
#절편 = Y평균 - (X평균 * 기울기)


from sklearn.linear_model import LinearRegression
x = np.array([2,4,6,8])
y = np.array([71,83,91,97])

x_mean = x.mean()
y_mean = y.mean()

d = sum([(i - x_mean)**2 for i in x])
d

def func(x,x_m,y,y_m):
    s = 0
    for i in range(len(x)):
        s += (x[i] - x_m) * (y[i] - y_m)
    return s

n = func(x,x_mean,y,y_mean)
n
a = n/d
a
b = y_mean - (x_mean * a)
b
print("기울기 : ",a)
print("절편 : ",b)
z = a * 3 + b
z

predict = a * x + b
error = predict - y

#평균제곱오차(Mean Squared Error)
#1/n * sum((predict - target)^2)

print("평균제곱오차 : ", (1/4) *(sum((predict - y)**2)))
print("평균제곱오차 : ", ((predict - y) **2).mean())

from sklearn.metrics import mean_squared_error
mean_squared_error(predict, y)

#평균제곱근오차(root mean square error)
def rmse(p,t):
    return np.sqrt(((p - t)**2).mean())

rmse(predict, y)
np.sqrt(mean_squared_error(predict, y))

#y = weight * x_data + bias
x_data = [1,2,3,4,5,6]
y_data = [2,4,6,8,10,12]

#weight와 bias값을 찾아야 한다.
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
w = tf.Variable(tf.random_normal([1], seed = 0, name = "weight"))
b = tf.Variable(tf.random_normal([1], seed = 0, name = 'bias'))

hypothesis = w * x + b
#오차 = Σ(hypothesis - y)²
cost = tf.reduce_mean(tf.square(hypothesis - y))

optimizer = tf.train.GradientDescentOptimizer(
        learning_rate = 0.001)#학습률, 조정을 잘 해야 한다. 
#cost값을 보면서 잘 조정해야 한다.
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_v, w_v, b_v, _ = sess.run([cost,w,b,train],
                                   feed_dict = {x:x_data,y:y_data})
    if (step % 20) == 0:
        print(step, cost_v, w_v, b_v)
        
sess.run(hypothesis, feed_dict = {x:3})
sess.run(hypothesis, feed_dict = {x:6})

###
x_data = [2,4,6,8]
y_data = [71,83,91,97]
###
data = [[2,71],[4,83],[6,91],[8,97]]
data1 = [i[0] for i in data]
data2 = [i[1] for i in data]
data1
data2

dir()

def GDO(X,Y,LR,N):
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    w = tf.Variable(tf.random_normal([1], seed = 0, name = "weight"))
    b = tf.Variable(tf.random_normal([1], seed = 0, name = 'bias'))
    hypothesis = w * x + b
    cost = tf.reduce_mean(tf.square(hypothesis - y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = LR)
    train = optimizer.minimize(cost)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for step in range(N+1):
        cost_v, w_v, b_v, _ = sess.run([cost,w,b,train],
                                   feed_dict = {x:X,y:Y})
        if (step % 20) == 0:
            print(step, cost_v, w_v, b_v)
    sess.close()
    
GDO(x_data, y_data,0.002,10000)
GDO([5,7,8,3], [2,3,6,4],0.002,10000)

#[문제] linear regression 학습을 통해서 입력값에 대한 예측값을 출력해주세요
#x1    x2    x3     y
#--------------------
#73    80    75     152
#93    88    93     185
#89    91    90     180
#96    98   100     196
#73    66    70     142
#y = w1 * x1 + w2 * x2 + w3 * x3 + b
#print("당신의 점수는 ",sess.run(hypothesis,feed_dict = {x1:100,x2:70,x3:60}))

def GDO2(x_data1, x_data2,x_data3,y_data,LR,N,X):
    x1 = tf.placeholder(tf.float32)
    x2 = tf.placeholder(tf.float32)
    x3 = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    w1 = tf.Variable(tf.random_normal([1], seed = 0, name = "weight1"))
    w2 = tf.Variable(tf.random_normal([1], seed = 0, name = "weight2"))
    w3 = tf.Variable(tf.random_normal([1], seed = 0, name = "weight3"))
    b = tf.Variable(tf.random_normal([1], seed = 0, name = 'bias'))
    hypothesis = w1 * x1 + w2 * x2 + w3 * x3 + b
    cost = tf.reduce_mean(tf.square(hypothesis - y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = LR)
    train = optimizer.minimize(cost)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for step in range(N+1):
        cost_v, w1_v,w2_v,w3_v, b_v, _ = sess.run([cost,w1,w2,w3,b,train],
                                                  feed_dict = {x1:x_data1,
                                                               x2:x_data2,
                                                               x3:x_data3,
                                                               y:y_data})
        
        if (step % 20) == 0:
            print(step, cost_v, w1_v,w2_v,w3_v, b_v)
            
    print("당신의 점수는 ",sess.run(hypothesis,feed_dict = {x1:X[0],
                                                      x2:X[1],x3:X[2]}))
    sess.close()

x_data1 = [73,93,89,96,73]
x_data2 = [80,88,91,98,66]
x_data3 = [75,93,90,100,70]
y_data = [152,185,180,196,142]
X = [100,70,60]
GDO2(x_data1, x_data2, x_data3, y_data, 0.000049, 10000,X)

## 선생님의 풀이 ##
#행렬을 배운것을 써 먹자 
x_data = [[73,80,75],[93,88,93],[89,91,90],[96,98,100],[73,66,70]]
y_data = [[152],[185],[180],[196],[142]]

x = tf.placeholder(tf.float32, shape = [None,3])#행의 갯수는 나중에 정함
y = tf.placeholder(tf.float32, shape = [None,1])
w = tf.Variable(tf.random_normal([3,1],seed = 1), name = 'weight')
b = tf.Variable(tf.random_normal([1],seed = 1), name = 'bias')

hypothesis = tf.matmul(x,w) + b
cost = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.0001)
train = optimizer.minimize(cost)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(100001):
    cost_v, hy_v, _ = sess.run([cost,hypothesis,train],
                                   feed_dict = {x:x_data,y:y_data})
        
    if (step % 100) == 0:
        print(step, cost_v, hy_v)
        
print("당신의 점수는 ",sess.run(hypothesis,feed_dict = {x:[[100,70,60]]}))
sess.close()
