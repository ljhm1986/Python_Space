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

##행렬 형태로 선언 
x = tf.placeholder(tf.float32, shape = (2,3))
print(x)
y = tf.placeholder(tf.float32, shape = (3,2))
print(y)
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
print(lin)
sess.run(lin)

#정수로 수열을 만드려면 
ran = tf.range(start = 1, limit = 10,
               delta = 1)#숫자의 간격
print(ran)
sess.run(ran)

ran = tf.range(11)
sess.run(ran)

#y = in * w + b
inputData = tf.Variable(tf.fill([1,32],7.))
print(inputData)
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
#reduce_mean() : 요소들의 평균값을 출력 
cost = tf.reduce_mean(tf.square(hypothesis - y))

optimizer = tf.train.GradientDescentOptimizer(
        learning_rate = 0.001)#학습률, 조정을 잘 해야 한다. 
#cost값을 보면서 잘 조정해야 한다.
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

#2000번 반복 시행 
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

##########################################################################
#11/22#
import tensorflow as tf
import numpy as np

#1~3열 독립변수, 4열 종속변수
data = np.loadtxt("C:\\WorkSpace\\Python_Space\\data\\ex.csv",
                  delimiter = ",",dtype = np.float32)

data

data[0][:-1]
x_data = data[:,:-1]
y_data = data[:,[-1]]
len(x_data)
len(x_data[0])
Z = [[80.0,70.0,60.0]]

outputData = []
def GDO3(X,Y,LR,N,Z):
    #X : 독립변수, Y : 종속변수, LR : 학습률, N : 학습횟수, Z : test data
    x = tf.placeholder(tf.float32, shape = [None,X.shape[1]])
    y = tf.placeholder(tf.float32, shape = [None,Y.shape[1]])
    w = tf.Variable(tf.random_normal([X.shape[1],Y.shape[1]],
                                      seed = 1, name = 'weight'))
    b = tf.Variable(tf.random_normal([Y.shape[1]], seed = 1, name = 'bias'))
    
    hypo = tf.matmul(x,w) + b
    cost = tf.reduce_mean(tf.square(hypo - y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = LR)
    train = optimizer.minimize(cost)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    for step in range(N+1):
        cost_v,hy_v,w_v,b_v,_ = sess.run([cost, hypo, w,b,train],
                                         feed_dict = {x:X,y:Y})
        
        if (step % 50) == 0:
            print("step : {}, cost : {},\n w : {}, b : {}"
                  .format(step, cost_v,w_v,b_v))
            print(hy_v)
            
            
    print("추정 점수는 ",sess.run(hypo,feed_dict = {x:Z}))
    sess.close()
    
GDO3(x_data, y_data, 0.00001, 10000,Z)

Z_1 = np.array(Z)
Z_1
w_1 = np.array([[0.7970418], [0.5342614], [0.6883417]])
b_1 = [-0.7836923]
np.dot(Z_1,w_1) + b_1 #=hypo , 마지막 추정점수와 동일함 
#y = w * x + b
#입력 : 1, 목표 : 4
#b : 1
#4 = w * 1 + 1 -> w = 3인데 
#다음과 같이 진행된다 (w값들을 대입하면서 )
#w = 1 -> y = 2
#w = 2 -> y = 3
#w = 2.5 -> y = 3.5
#w = 3 -> y = 4, cost = 0
#계속 조정하면서 계산을 한다.

#bias = ?, x = 1, weight = 2, y = 4
4 == 2 * 1 + 2

#logistic regression - 이진분류
#- 분류를 하는데 있어서 가장 흔한 경우는 이분법을 기준으로 분류
#- binary classification

x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]
y_data = [[0],[0],[0],[1],[1],[1]]

x = tf.placeholder(tf.float32, shape = [None, 2])
y = tf.placeholder(tf.float32, shape = [None, 1])

w = tf.Variable(tf.random_normal([2,1], name = 'weight'))
b = tf.Variable(tf.random_normal([1], name = 'bias'))

hypothesis = tf.matmul(x,w) + b
#지금까지는 hypothesis에 활성화함수를 사용하지 않았다. 예측값 그대로 사용했다.
#이를 0 또는 1 로 바꾸어야 한다. 
hypothesis = tf.sigmoid(tf.matmul(x,w) + b)
#그럼 cost값은 어떻게 계산해야 하나, 기존의 공식으로는 안되겠다.
#
#시그모이드 함수를 어떻게 해야 (그래프 형태가) 2차식처럼 
#볼록하게 할수 있을까? 기존의 오차함수와 다른 형태일 것이다.

#예측 1, 실제 0, log0 = -무한 
#크로스 엔트로피 오차 함수를 만들어서 사용해야 한다.
 
#y = 1   -y * log(h(x))
#y = 0   (1 - y) * log(1-h(x))
#그럼 위 두 식을 합치면 
#y * tf.log(hypothesis) + (1 - y) * tf.log(1 - hypothesis)
#
cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1 - hypothesis))

###


###
train = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)
#0.5 이상 -> 1, 미만 -> 0 표현
predict = tf.cast(hypothesis > 0.5, dtype = tf.float32)

#cast() 는 다음과 같이 작동한다. 
#if hypothesis > 0.5 then:
#    True(1,0)
#else:
#    False(0.0)

#같은지 비교하자 
#equal() 같으면 true, 다르면 false
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, y), dtype = tf.float32))
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(10001):
    cost_val, _ = sess.run([cost, train],
                           feed_dict = {x:x_data, y:y_data})
    
    if step % 1000 == 0:
        print(step, cost_val)
        
h,p,a = sess.run([hypothesis, predict, accuracy],
                 feed_dict={x:x_data, y:y_data})

print("hypothesis : ",h)
print("predict : ",p)
print("accurary : ",a)

sess.run(hypothesis, feed_dict = {x:[[7,3]]})
sess.run(hypothesis, feed_dict = {x:[[1,3]]})
tf.cast( sess.run(hypothesis, feed_dict = {x:[[7,3]]}) > 0.5, dtype = tf.int32)


#[문제] xor를 logistic regression classifier 를 이용해서 프로그램 생성하세요
#0 0 0
#0 1 1
#1 0 1
#1 1 0



x_data = [[0,0],[0,1],[1,0],[1,1]]
y_data1 = [[0],[1],[1],[1]]
y_data2 = [[1],[1],[1],[0]]
y_data3 = [[0],[1],[1],[0]]

def LRC(X,Y,LR,N,Z):
    #X : 독립변수, Y : 종속변수, LR : 학습률, N : 학습횟수, Z : test data
    x = tf.placeholder(tf.float32, shape = [None, 2])
    y = tf.placeholder(tf.float32, shape = [None, 1])

    w = tf.Variable(tf.random_normal([2,1], name = 'weight'))
    b = tf.Variable(tf.random_normal([1], name = 'bias'))

    hypothesis = tf.sigmoid(tf.matmul(x,w) + b)
    cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1 - hypothesis))
    train = tf.train.GradientDescentOptimizer(learning_rate = LR).minimize(cost)

    predict = tf.cast(hypothesis > 0.5, dtype = tf.float32)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, y), dtype = tf.float32))
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    for step in range(N+1):
        cost_val, w_v, b_v,_ = sess.run([cost, w,b,train],
                                        feed_dict = {x:X, y:Y})
    
        if step % 1000 == 0:
            print(step, cost_val, w_v, b_v)
            
    h,p,a = sess.run([hypothesis, predict, accuracy],feed_dict={x:X, y:Y})

    print("hypothesis : ",h)
    print("predict : ",p)
    print("accurary : ",a)

    pro = tf.cast(sess.run(hypothesis,
                          feed_dict = {x:Z}) > 0.5, dtype = tf.int32) 
    print(sess.run(pro))
    sess.close()
    return sess.run(pro)
    

y_data = [[0],[1],[1],[0]]
z = x_data
def LRC_XOR(x_data,y_data1,y_data2,y_data3,0.01,10000,z):
    temp1 = LRC(x_data,y_data1,0.01,10000,z)s
    print(type(temp1))
    temp2 = LRC(x_data,y_data2,0.01,10000,z)
    print(type(temp2))
    
#음 별로다...
    
## 선생님의 풀이 ##
x_data = [[0,0],[0,1],[1,0],[1,1]]
y_data = [[0],[1],[1],[0]]
x_data = np.array(x_data, dtype = np.float32)
x_data.shape   
y_data = np.array(y_data, dtype = np.float32) 
y_data.shape

x = tf.placeholder(tf.float32, shape = [None, 2])
y = tf.placeholder(tf.float32, shape = [None, 1])

w1 = tf.Variable(tf.random_normal([2,5], seed = 1), name = 'weight1')
b1 = tf.Variable(tf.random_normal([5], seed = 1), name = 'bias1')#(5,)
b1.shape # b1 = [b1,b2,b3,b4,b5]
#나중에 b1 더할때 브로드캐스팅이 된다.
layer1 = tf.sigmoid(tf.matmul(x,w1) + b1) #(None, 5)
print(layer1) 

w2 = tf.Variable(tf.random_normal([5,4], seed = 1), name = 'weight2')
b2 = tf.Variable(tf.random_normal([4], seed = 1), name = 'bias2')
layer2 = tf.sigmoid(tf.matmul(layer1,w2) + b2) #(None, 4)

w3 = tf.Variable(tf.random_normal([4,4], seed = 1), name = 'weight3')
b3 = tf.Variable(tf.random_normal([4], seed = 1), name = 'bias3')
layer3 = tf.sigmoid(tf.matmul(layer2,w3) + b3) #(None, 4)

w4 = tf.Variable(tf.random_normal([4,1], seed = 1), name = 'weight4')
b4 = tf.Variable(tf.random_normal([1], seed = 1), name = 'bias4')
hypothesis = tf.sigmoid(tf.matmul(layer3,w4) + b4)

cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1-y) * tf.log(1 - hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate = 0.05).minimize(cost)

predict = tf.cast(hypothesis > 0.5, dtype = tf.float32)

accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, y), dtype = tf.float32))
sess = tf.Session()
sess.run(tf.global_variables_initializer())
    
for step in range(20001):
    cost_val, w_v, b_v,_ = sess.run([cost, w,b,train],
                                        feed_dict = {x:x_data, y:y_data})
    
    if step % 500 == 0:
        print(step, cost_val, w_v, b_v)

pro = tf.cast(sess.run(
        hypothesis,
        feed_dict = {x:[[0,0],[0,1],[1,0],[1,1]]}) > 0.5,
        dtype = tf.int32) 
print(sess.run(pro))
sess.close()

#######
def layer(x,n):
    w = tf.Variable(tf.random_normal([x.shape[1],n], seed = 1), name = 'weight1')
    b = tf.Variable(tf.random_normal([n], seed = 1), name = 'bias1')
    layer = tf.sigmoid(tf.matmul(x,w) + b)
    
    return layer 
    
def XOR(X,Y,LR,N,Z):
    #X : 독립변수, Y : 종속변수, LR : 학습률, N : 학습횟수, Z : test data

    x = tf.placeholder(tf.float32, shape = [None, X.shape[1]])
    y = tf.placeholder(tf.float32, shape = [None, Y.shape[1]])

    lay1 = layer(x_data,5)
    #lay2 = layer(lay1,4)#음 꼭 레이어를 4개 있을 필요는 없어 보인다. 
    #lay3 = layer(lay2,4)
    hypo = layer(lay1,1)#layer 가 1개이면 cost가 어느 시점부터 줄어들지 않는다.

    cost = -tf.reduce_mean(y * tf.log(hypo) + (1-y) * tf.log(1 - hypo))

    train = tf.train.GradientDescentOptimizer(learning_rate = LR).minimize(cost)
    
    predict = tf.cast(hypothesis > 0.5, dtype = tf.float32)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, y), dtype = tf.float32))
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    for step in range(N+1):
        cost_val, w_v, b_v,_ = sess.run([cost, w,b,train],
                                            feed_dict = {x:x_data, y:y_data})
    
        if step % 500 == 0:
            print(step, cost_val, w_v, b_v)

    pro = tf.cast(sess.run(hypo,feed_dict = {x:Z}) > 0.5,dtype = tf.int32) 
    print(sess.run(pro))
    sess.close()

x_data
y_data
Z = [[0,0],[0,1],[1,0],[1,1]]

XOR(x_data, y_data,0.1,30000,Z)

#############################################################################
#11/25#
import numpy as np
import pandas as pd
import tensorflow as tf

np.exp(1)
np.exp(10000) # inf, 저장할 공간이 부족하다. 
type(np.exp(10000))
np.exp(10)

x = np.array([10,6,5])
np.exp(x) / np.sum(np.exp(x))
np.sum(np.exp(x) / np.sum(np.exp(x))) == 1

x = np.array([1010,1000,990])
np.exp(x) / np.sum(np.exp(x))
#이런 경우에는 모든 수에 최대값으로 빼주자 
m = np.max(x)
y = x - m
np.exp(y) / np.sum(np.exp(y))

def softmaxfunction(x):
    max_x = np.max(x)
    soft = np.exp(x - max_x) / np.sum(np.exp(x - max_x))
    return soft

softmaxfunction(x)
np.sum(softmaxfunction(x))

x = np.array([1010, 1000, 990])
s = softmaxfunction(x)
np.sum(s)
np.argmax(s)
s
x = np.array([33,5,9,20000])

#종속변수를 표시할 때 .. 이진분류때와 다르다 
#
xy = np.loadtxt("c:/WorkSpace/Python_Space/data/train.txt",
                delimiter = ',', dtype = np.float32)

xy
x_data = xy[:,:-1]
y_data = xy[:,[-1]]
y_data.shape# (8,1)
y_dataa = xy[:,-1]
y_dataa.shape # (8,) 이러면 안 됨 

x = tf.placeholder(tf.float32, shape = [None, 3])
y = tf.placeholder(tf.int32, shape = [None, 1])
y.shape
#TensorShape([Dimension(None), Dimension(1)])

#one hot 인코딩은 항상 정수로만 이루어진다. 
y_one_hot = tf.one_hot(y,3) #one-hot 텐서 반환, column의 갯수가 3 
y_one_hot
print(y_one_hot)
#Tensor("one_hot_8:0", shape=(?, 1, 3), dtype=float32)
y_one_hot = tf.reshape(y_one_hot, [-1,3])
print(y_one_hot)
#Tensor("Reshape_8:0", shape=(?, 3), dtype=float32)
w = tf.Variable(tf.random_normal([3,3]), name = 'weight')
b = tf.Variable(tf.random_normal([3]), name = 'bias')

logits = tf.matmul(x,w) + b
hypothesis = tf.nn.softmax(logits)

#cost는?? 
cost_i = tf.nn.softmax_cross_entropy_with_logits(
        logits = logits, labels = y_one_hot)
cost = tf.reduce_mean(cost_i)

train = tf.train.GradientDescentOptimizer(
        learning_rate = 0.01).minimize(cost)

#cost에 대해서 살펴보자 
#cross entropy function
# - Σ (Y * log(Y_))
#f(Y_) =  Σ (Y * -log(Y_)) 로 보면 
#     0
#Y = [ ]
#     1
#
# 0,1 이 나와야 하는데 0,1이 나왔을 때 f() 값은
#      0 : 0     INF = 0
#Y_ =  1 : 1  *   0  = 0  
# 
# 0,1 이 나와야 하는데 1,0이 나왔을 떄 f() 값은
#     1 : 0    0  = 0
#Y_ = 0 : 1 * INF = INF
#cost 가  INF 이니까 다시 계산한다... 
#이러한 작업을 softmax_cross_entropy_with_logits가 대신 해 주었다.

#argmax() : 최대값인 요소의 번호를 반환함 
predict = tf.argmax(hypothesis, 1)#0 : 축, 1 : 횡
#0 이면 같은 column에 있는 원소들 중에서 고름,
#1 이면 같은 row에 있는 원소들 중에서 고름
correct_predict = tf.equal(predict, tf.argmax(y_one_hot,1))

accuracy = tf.reduce_mean(tf.cast(correct_predict, dtype = tf.float32))
#일치하면 1, 다르면 0일 것이고 거기에 평균을 내니 맞추는 비율이 나온다. 

sess = tf.Session()
sess.run(tf.global_variables_initializer())
    
for step in range(10001):
    cost_val,w_v,b_v,_ = sess.run([cost, w,b,train],
                          feed_dict = {x:x_data, y:y_data})
    
    if step % 500 == 0:
        print(step, cost_val)
        
loss, acc, hyp, logi = sess.run([cost, accuracy, hypothesis, logits],
                                feed_dict ={x:x_data, y:y_data})
print(loss, acc, hyp, logi)

a = sess.run(hypothesis, feed_dict = {x: [[1,2,1]]})
print(a, sess.run(tf.argmax(a,1)))
b = sess.run(hypothesis, feed_dict = {x: [[1,7,7]]})
print(b, sess.run(tf.argmax(b,1)))

#학습된 신경망에 본래 x_data를 넣어보고 출력값을 비교해보자 
score = sess.run(hypothesis, feed_dict = {x:x_data})
print(score)
print(sess.run(tf.argmax(score,1)))
(sess.run(tf.argmax(score,1)).reshape(8,1) == y_data).sum() / 0.08

###
target = [1,0,0]
h = [0.7, 0.2, 0.0]

np.sum(target * - np.log(h))
#h의 성분중 0 이 있으면 nan이 된다. 

def cross_entropy_function(h,t):
    h = np.array(h)
    t = np.array(t)
    #nan이 되는것을 방지하기 위해 작은 값을 넣자 
    delta = 1e-7 
    return -np.sum(t * np.log(h + delta))

cross_entropy_function(h,target)

#####
bmi = pd.read_csv("C:/WorkSpace/Python_Space/data/bmi.csv")
bmi
bmi['label'].unique()
#fat, normal, thin 3개가 있다 

y_data = [0 if i == 'fat' else 1 if i == 'normal' else 2 for i in bmi['label']]
y_data
y_data = np.array(y_data)
y_data.shape
y_data = np.atleast_2d(y_data)
y_data = np.transpose(y_data)
y_data.shape

x_data = bmi.iloc[:,:-1]
x_data.shape

x = tf.placeholder(tf.float32, shape = [None, 2])
y = tf.placeholder(tf.int32, shape = [None, 1])

y_one_hot = tf.one_hot(y,3)
y_one_hot = tf.reshape(y_one_hot, [-1,3])

w1 = tf.Variable(tf.random_normal([2,4]), name = 'weight1')
b1 = tf.Variable(tf.random_normal([4]), name = 'bias1')
layer1 = tf.matmul(x, w1) + b1

w2 = tf.Variable(tf.random_normal([4,3]), name = 'weight2')
b2 = tf.Variable(tf.random_normal([3]), name = 'bias2')

logits = tf.matmul(layer1,w2) + b2
hypothesis = tf.nn.softmax(logits)

cost_i = tf.nn.softmax_cross_entropy_with_logits(
        logits = logits, labels = y_one_hot)
cost = tf.reduce_mean(cost_i)

train = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)

predict = tf.argmax(hypothesis, 1)
correct_predict = tf.equal(predict, tf.argmax(y_one_hot,1))
accuracy = tf.reduce_mean(tf.cast(correct_predict, dtype = tf.float32))
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(20001):
    cost_val,w_v,b_v,_ = sess.run([cost, w,b,train],
                          feed_dict = {x:x_data, y:y_data})
    
    if step % 500 == 0:
        print(step, cost_val, w_v, b_v)
        
        
a = sess.run(hypothesis, feed_dict = {x: [[150,40]]})
print(a, sess.run(tf.argmax(a,1)))

xx = sess.run(hypothesis, feed_dict = {x:x_data})
(sess.run(tf.argmax(xx,1)).reshape(20000,1) == y_data).sum() / 200

#이번에는 단위가 동일하지 않았다. 그럼 스케일링 작업을 하고 
#훈련을 하는게 좋겠다. 
### 선생님의 풀이 ###
import pandas as pd
bmi.info()
bmi['label']

set(bmi['label'])
bmi.loc[bmi.label == 'thin', 'label'] = 0
bmi.loc[bmi.label == 'normal', 'label'] = 1
bmi.loc[bmi.label == 'fat', 'label'] = 2
bmi['label']

x_data = bmi.iloc[:,0:2]
y_data = bmi.iloc[:,[2]]
x = tf.placeholder(tf.float32, [None,2])
y = tf.placeholder(tf.int32, [None,1])

y_one_hot = tf.one_hot(y, 3)
y_one_hot = tf.reshape(y_one_hot, [-1,3])

w = tf.Variable(tf.random_normal([2,3],seed = 1), name = 'weight')
b = tf.Variable(tf.random_normal([3], seed = 1), name = 'bias')

logits = tf.matmul(x,w) + b
hypothesis = tf.nn.softmax(logits)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits = logits, labels = y_one_hot))

train = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)

predict = tf.argmax(hypothesis,1)
correct_predict = tf.equal(predict, tf.argmax(y_one_hot,1))
accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(10001):
    sess.run(train, feed_dict = {x:x_data, y:y_data})
    
    if step % 1000 == 0:
        loss, acc = sess.run([cost, accuracy],
                             feed_dict = {x:x_data, y:y_data})
        print("step{:5}\tloss:{:.3f}\tAcc{:.1%}".format(step,loss,acc))
    
#여기까지 보면 accuracy가 약 70% 정도 나온다.
#scale 작업을 해 보자         
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_scale = scaler.fit_transform(x_data)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(10001):
    sess.run(train, feed_dict = {x:x_scale, y:y_data})
    
    if step % 1000 == 0:
        loss, acc = sess.run([cost, accuracy],
                             feed_dict = {x:x_scale, y:y_data})
        print("step{:5}\tloss:{:.3f}\tAcc{:.1%}".format(step,loss,acc))
        
#accuracy가 약 98% 나온다. 성공!!
test_h = (166 - np.mean(bmi['height'])) / np.std(bmi['height'])
test_w = (58 - np.mean(bmi['weight'])) / np.std(bmi['weight'])

a = sess.run(hypothesis, feed_dict = {x:[[test_h,test_w]]})
print(a, sess.run(tf.argmax(a,1)))

#minmax로 해 보자 
from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()
x_minmax = minmax.fit_transform(x_data)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(10001):
    sess.run(train, feed_dict = {x:x_minmax, y:y_data})
    
    if step % 1000 == 0:
        loss, acc = sess.run([cost, accuracy],
                             feed_dict = {x:x_minmax, y:y_data})
        print("step{:5}\tloss:{:.3f}\tAcc{:.1%}".format(step,loss,acc))
#accuracy가 약 90% 나온다. standard 보다 낮은 수치이다.
        
test_h = (166 - bmi['height'].min()) / (bmi['height'].max() - bmi['height'].min())
test_w = (68 - bmi['weight'].min()) / (bmi['weight'].max() - bmi['weight'].min())

a = sess.run(hypothesis, feed_dict = {x:[[test_h,test_w]]})
print(a, sess.run(tf.argmax(a,1)))

