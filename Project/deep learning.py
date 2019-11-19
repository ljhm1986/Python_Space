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

############################################################################
#11/19#

#numpy 
#다차원 배열 객체를 제공
#과학적 계산, 데이터 분석을 이용할 때 사용하는 라이브러리
import numpy as np

a = np.array([1,2,3])
a
#array([1, 2, 3])
type(a)#numpy.ndarray
a.shape
#(3,)
a.ndim
#1
a.dtype
#dtype('int32')

a[0]
a[0] = 4 #수정하기
a
#array([4, 2, 3])

b = np.array([[1,2,3],[4,5,6]])
b
#array([[1, 2, 3],
#       [4, 5, 6]])
print(b)
#[[1 2 3]
# [4 5 6]]
b.shape
b.ndim
b.dtype
b[0]
b[0,0]
b[0][0]
b[:,0]

#0으로만 채워져 있는 
np.zeros((2,2))
np.zeros((4,5))
#1로만 채워져 있는
np.ones((2,2))

#해당 숫자로만 채워져 있는 
np.full((2,2),7)
np.full((4,3),5)

#대각 행렬
np.eye(2)
np.eye(4)
#임의의 값으로 채우기
np.random.random((2,2))
np.random.random((3,3))
np.empty((2,2))
np.empty((3,4))
np.empty((5,6))

#list -> array 로 바꾸기
data1 = [1,2,3,4,5]
type(data1)
data2 = np.array(data1)
type(data2)
data2.shape
data2.ndim
data2.dtype

data3 = [[1,2,3],[4,5,6]]
type(data3)
data4 = np.array(data3)
type(data4)

np.zeros((3,4))
np.zeros((2,3,4)) #면, 행, 열 : 3차원

np.zeros((4,3,2))
np.full((4,6,5),2)

np.random.random((4,3,2))

#인덱싱 
x = np.array(([[1,2,3,4],[5,6,7,8],[9,10,11,12]]))
x
x.shape
x[0]
x[0,0]
x[:,2]
x[2,:]
x[0,0:2]
x[0:2,1]
x[0:2,2]

##요소 추출하기 
x = np.array([[1,2],[3,4],[5,6]])

#1,4,5 를 추출해보자
x[0,0],x[1,1],x[2,0]
#한 번에 추출하기 
x[[0,1,2],[0,1,0]] #[[행],[열]]
np.array([x[0,0],x[1,1],x[2,0]])

x[[0,0],[1,1]] #array
[x[0,1],x[0,1]] #list

x = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
#1 6 7 11 추출하기
x[[0,1,2,3],[0,2,0,1]]
np.array([x[0,0],x[1,2],x[2,0],x[3,1]])

b = [0,2,0,1]
x[np.arange(4),b]

#
x[[1,2],[0,1]]

#모든 요소에 10을 더하기
x[np.arange(4),b] += 10
x[np.arange(4),b]
x
#array([[11,  2,  3],
#       [ 4,  5, 16],
#       [17,  8,  9],
#       [10, 21, 12]])

x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])

#요소별 합
x + y
np.add(x,y)
#요소별 차
x - y
np.subtract(x,y)
#요소별 곱
x * y
np.multiply(x,y)
#요소별 나누기
x/y
np.divide(x,y)

#요소별 제곱근
np.sqrt(x)
np.sqrt(y)

v = np.array([9,10])
w = np.array([11,12])
#행렬의 곱
v.dot(w)
np.dot(v,w)

np.dot(x,v)
np.dot(y,w)

#모든 요소의 합
np.sum(x)
np.sum(y)
np.sum(v)
np.sum(w)

print(x)
#[[1 2]
# [3 4]]
#행을 열 기준으로 합치기
np.sum(x, axis = 0)
#array([4, 6])
#열을 행 기준으로 합치기
np.sum(x, axis = 1)
#array([3, 7])

#각 열의 평균 구하기
np.mean(x, axis = 0)
#array([2., 3.])
#각 행의 평균 구하기 
np.mean(x, axis = 1)
#array([1.5, 3.5])

#분산
np.var(x)
np.var(y)

#표준편차
np.std(x)
np.std(y)

#누적합
np.cumsum(x)
#행 방향 누적합
np.cumsum(x, axis = 0)
#열 방향 누적합
np.cumsum(x, axis = 1)

#요소 곱
np.prod(x)
np.prod(x, axis = 0)
np.prod(x, axis = 1)

np.prod(x, keepdims = True)#결과가 숫자가 아니라 1 * 1 array로 나온다.
#array([[24]]) 

np.prod(x)

x = np.array([[1,2],[3,np.nan]])

np.sum(x)
#nan
np.nansum(x)
#6.0

np.prod(x)
#nan
np.nanprod(x)
#6.0

x
np.max(x)
np.max(x, axis = 0)
#array([ 3., nan])
np.nanmax(x, axis = 1)
#array([2., 3.])

#누적곱
np.cumprod(x)
#array([ 1.,  2.,  6., nan])
np.nancumprod(x)
#array([1., 2., 6., 6.])
np.nancumprod(x, axis = 0)
#array([[1., 2.],
#       [3., 2.]])
np.nancumprod(x, axis = 1)
#array([[1., 2.],
#       [3., 3.]])

#브로드캐스팅
#shape이 다른 배열간에도 산술연산이 가능하게 하는 메커니즘 
x = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
y = np.array([1,0,1])

x + y
for i in x:
    print(i + y)
    
z = np.zeros((4,3))
z

#모든 요소가 0이고 x와 shape가 같은 z만들기
z = np.zeros_like(x)
z = np.zeros(x.shape)

x.shape[0]

#브로드캐스팅 처럼 하려면 
for i in range(x.shape[0]):
    z[i,:] = x[i,:] + y
    
z

#y를 4번 쌓기
z = np.tile(y,(4,1))
print(y,z)

#
x = np.array([2,3,4])
y = np.array([1,5,2])
#요소별 최대값
np.maximum(x,y)
#요소별 최소값
np.minimum(x,y)

#최대값이 있는 인덱스 번호
np.argmax(x)
#최소값이 있는 인덱스 번호
np.argmin(x)

x[np.argmax(x)]
x[np.argmin(x)]

x = np.array([[1,8,2],[3,2,5],[9,5,3]])
print(x)
#[[1 8 2]
# [3 2 5]
# [9 5 3]]

np.argmax(x)
np.argmin(x)

#열별 최대값이 있는 인덱스 번호
np.argmax(x, axis = 0)
#array([2, 0, 1], dtype=int64)

#행별 최대값이 있는 인덱스 번호
np.argmax(x, axis = 1)
#array([1, 2, 0], dtype=int64)

#열별 최소값이 있는 인덱스 번호
np.argmin(x, axis = 0)
#array([0, 1, 0], dtype=int64)

#행별 최소값이 있는 인덱스 번호
np.argmin(x, axis = 1)
#array([0, 1, 2], dtype=int64)

name = np.array(['a','b','a','b','c'])
np.unique(name)

ids = np.array([1,2,4,2,6,8,9,4,2,2,54,1,5,7,7,2,2,4,8,90,4,4,2])
np.unique(ids)
set(ids)
sorted(set(ids))

ids == 1
np.in1d(ids,[1])

x = np.array(['a','b','a','b','c','c'])
y = np.array(['a','b','a','b','d','b'])

#교집합
np.intersect1d(x,y)

#합집합
np.union1d(x,y)

#차집합
np.setdiff1d(x,y)

x = np.array([1,2,4,10,15,20])

#배열 원소간 차 구하기 
np.diff(x)
#array([1, 2, 6, 5, 5])
np.diff(x, n = 2)
#array([ 1,  4, -1,  0])
np.diff(x, n = 3)
#array([ 3, -5,  1])

y = np.array([[1,2,3,4],[5,6,7,8]])
np.diff(y)
#array([[1, 1, 1],
#       [1, 1, 1]])
np.diff(y, axis = 0)
#array([[4, 4, 4, 4]])
np.diff(y, axis =1)
#array([[1, 1, 1],
#       [1, 1, 1]])
np.diff(x,y,n=2)# 안 되넹
np.diff([x,y], n = 2) #안 되넹


#np 배열의 저장 
x = np.arange(10)
y = np.arange(100)

np.save("c:/WorkSpace/Python_Space/npfile/x.npy",x)
x2 = np.load("c:/WorkSpace/Python_Space/npfile/x.npy")
x2

np.savez("c:/WorkSpace/Python_Space/npfile/xy.npz",x = x,y = y)
z = np.load("c:/WorkSpace/Python_Space/npfile/xy.npz")
z#객체값 출력
z['x']
#array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
z['y']
#array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
#       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
#       34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
#       51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
#       68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
#       85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99])

#저장할때 확장자의 이름을 잘 모르겠으면 .확장자 를 빼고 써도 된다.
np.save("c:/WorkSpace/Python_Space/npfile/x1",x)
np.savez("c:/WorkSpace/Python_Space/npfile/xy1",x = x,y = y)

#일반 연산의 속도와 numpy 연산 속도를 비교해보자
import time

size = 1000000

def original():
    start = time.time()
    x = range(size)
    y = range(size)
    z = []
    
    for i in range(len(x)):
        z.append(x[i] + y[i])
    
    end = time.time()
    
    return end - start

def numpy():
    start = time.time()
    x = np.arange(size)
    y = np.arange(size)
    z = x + y
    end = time.time()
    return end - start

t1 = original()
t2 = numpy()

print("original process time : %.5fs"%t1)
print("numpy process time : %.5fs"%t2)
#행렬을 사용해서 연산이 빠르다. 

#transpose
x = np.array([[1,2],[3,4]])
x.T
x

x = np.arange(10)
x
x.T
x.reshape((5,2))
x.reshape((5,2), order = 'C') #행 우선
x.reshape((5,2), order = 'F') #열 우선

y = np.arange(10).reshape((5,2))
y
#array([[0, 1],
#       [2, 3],
#       [4, 5],
#       [6, 7],
#       [8, 9]])


#펼치기
y.reshape(10,)
y.ravel()
y.ravel('C') #행 순서로 나열 
#array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
y.ravel('F') #열 순서로 나열
#array([0, 2, 4, 6, 8, 1, 3, 5, 7, 9])
y.flatten()

#행렬 붙이기 
x = np.array([[1,2,3],[4,5,6]])
y = np.array([[7,8,9],[10,11,12]])

np.concatenate([x,y], axis = 0)# 행으로 붙이기
#array([[ 1,  2,  3],
#       [ 4,  5,  6],
#       [ 7,  8,  9],
#       [10, 11, 12]])
np.concatenate([x,y], axis = 1)# 열로 붙이기
#array([[ 1,  2,  3,  7,  8,  9],
#       [ 4,  5,  6, 10, 11, 12]])

np.vstack([x,y])
#array([[ 1,  2,  3],
#       [ 4,  5,  6],
#       [ 7,  8,  9],
#       [10, 11, 12]])
np.hstack([x,y])
#array([[ 1,  2,  3,  7,  8,  9],
#       [ 4,  5,  6, 10, 11, 12]])

#반복하기 
x = np.arange(3)
x
#array([0, 1, 2])
x.repeat(2)
#array([0, 0, 1, 1, 2, 2])
x.repeat([2,3,4])#요소 별로 반복 횟수를 다르게 지정한다.
#array([0, 0, 1, 1, 1, 2, 2, 2, 2])

#
z = np.random.randn(2,2)
z
z.repeat(2, axis = 0)
z.repeat(2, axis = 1)
z.repeat([2,3], axis = 1)
np.tile(z,2)

s = np.array([5,3,4,2,1])
s
s.argsort() #sort한 이후의 인덱스 번호 
#array([4, 3, 1, 2, 0], dtype=int64)
s[s.argsort()]
#array([1, 2, 3, 4, 5])

np.argsort(s) #sort한 이후의 인덱스 번호 , 오름차순
#array([4, 3, 1, 2, 0], dtype=int64)
np.argsort(-s) #sort한 이후의 내림차순 번호, 내림차순
#array([0, 2, 1, 3, 4], dtype=int64)

np.sort(s)#오름차순
#array([1, 2, 3, 4, 5])
np.sort(s)[::-1]#내림차순
#array([5, 4, 3, 2, 1])

s1 = np.array([[3,1,7],[2,8,4],[1,0,3]])
np.sort(s1)
np.sort(s1, axis = 1)#각 행 정렬
np.sort(s1, axis = 0)#각 열 정렬 

id(s1) #메모리 주소 확인
s2 = s1
id(s2) #같다!!!

x = np.array([0,1,2,3])
id(x)
y = x
id(y) #같다!!

x[0]
y[0] = 10
x[0]

#메모리를 다르게 복사해야 한다.
z = x.copy()
id(z)

x[0] = 100
x
y
z

#list로 바꾸기 
type(z)
z = z.tolist()
type(z)

x = np.array([1,4,6,8,10,13,16,20])
#15를 어디어 넣는게 좋을까?

x.searchsorted(15)
#6

y = np.array([1,4,6,16,10,13,8,20])
y.searchsorted(15)
#7
#정렬이 되어 있는 경우에 쓰는게 일반적이다.

np.insert(x,6,15)#x의 6번 index에 15를 넣기
x

a = np.arange(1,7).reshape(3,2)
a
np.insert(a,3,[7,8], axis = 0)
np.insert(a,1,[7],axis = 0)
#array([[1, 2],
#       [7, 7],
#       [3, 4],
#       [5, 6]])
np.insert(a,1,[9],axis = 1)
#array([[1, 9, 2],
#       [3, 9, 4],
#       [5, 9, 6]])


np.delete(a,1,axis = 0) #2행 삭제
np.delete(a,1,axis = 1) #2열 삭제

# 1 2 3      5 6
#(     ) * ( 7 8 ) = ?
# 4 5 6      9 10

#a * b x * y

a = np.array([[1,2,3],[4,5,6]])
b = np.array([[5,6],[7,8],[9,10]])
a.dot(b)
np.dot(a,b)
a.shape
b.shape

z = np.array([5,6,7])
np.dot(a,z)

